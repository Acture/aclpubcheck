"""Where aclpubcheck spends its time, and how this checkout compares with a base version.

    python benchmarks/bench_checks.py --label local-synthetic
    python benchmarks/bench_checks.py --label local-sigdial --papers-yml PATH/papers.yml \
        --base-python PATH/TO/main/venv/bin/python --base-label main

Measures on a generated corpus of synthetic papers by default, or on real PDFs (--papers-yml,
--pdf-dir). Prints a table and writes benchmarks/results/<label>.json. CI runs it on the
SIGDIAL example; the results committed here come from that run.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pdfplumber
import yaml

REPO = Path(__file__).resolve().parent.parent
# measure this checkout's aclpubcheck; the synthetic papers come from the test fixtures
sys.path[:0] = [str(REPO), str(REPO / "tests")]

from pdf_fixtures import MARGIN_TEXT, REFERENCES, TEXT, write_pages

from aclpubcheck.batch.check import CheckJob, run_check
from aclpubcheck.batch.console import ConsoleSink
from aclpubcheck.batch.manifest import LocalPdfProvider, load_papers_yml
from aclpubcheck.batch.model import (
    Event,
    Finding,
    PaperChanged,
    PaperRecord,
    PaperResult,
    Status,
    fan_out,
)
from aclpubcheck.batch.runner import RunOptions, run_batch
from aclpubcheck.formatchecker import Formatter

PACKAGES = ("aclpubcheck", "pdfplumber", "pdfminer.six", "pypdfium2", "numpy", "pillow")
IMPORT_PROBE = (
    "import time; t = time.perf_counter(); import aclpubcheck.formatchecker; "
    "print(time.perf_counter() - t)"
)
# a fresh interpreter, so neither the timing nor the peak memory includes the benchmark's own
NAME_CHECK_PROBE = """
import json, sys, time
from aclpubcheck.name_check import PDFNameCheck
started = time.perf_counter()
PDFNameCheck()
seconds = time.perf_counter() - started
peak = None
if sys.platform != "win32":
    import resource
    scale = 2**20 if sys.platform == "darwin" else 2**10  # ru_maxrss: bytes on macOS, else KiB
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / scale
print(json.dumps({"seconds": seconds, "peak_rss_mb": peak}))
"""
FORMATTER_REPEAT = 1000

# status, errors, warnings, message: what must not depend on how the papers were scheduled
Semantic = tuple[Status, tuple[Finding, ...], tuple[Finding, ...], str]


@dataclass(frozen=True)
class Kind:
    pages: tuple[bytes, ...]
    expected: Status


# synthetic papers; all are long papers, so the 11-page one breaks the page limit
KINDS = {
    "clean": Kind((TEXT,), Status.PASSED),
    "margin": Kind((MARGIN_TEXT,), Status.VIOLATIONS),
    "pagelimit": Kind((TEXT,) * 10 + (REFERENCES,), Status.VIOLATIONS),
}


@dataclass(frozen=True)
class Corpus:
    source: str  # "generated", or the papers.yml / PDF directory measured
    manifest: Path
    papers_dir: Path
    kinds: dict[str, str]  # paper id -> synthetic kind; empty for real papers


@dataclass(frozen=True)
class Stats:
    min: float
    median: float
    max: float

    @classmethod
    def of(cls, values: Sequence[float]) -> Stats:
        return cls(min(values), statistics.median(values), max(values))


@dataclass(frozen=True)
class PaperTiming:
    paper_id: str
    kind: str
    pages: int | None  # None for a check_error paper, whose PDF may not open
    seconds: float
    status: str


@dataclass(frozen=True)
class BatchRun:
    workers: int
    repeat: int
    wall_s: float
    checked: int  # papers that reached the format check
    first_result_s: float | None  # pool start-up latency, as seen by the first checked paper
    busy_s: float  # summed per-paper fetch+check time; busy_s / wall_s is the concurrency achieved

    @property
    def papers_per_s(self) -> float:
        return self.checked / self.wall_s


@dataclass(frozen=True)
class PathModeRun:
    """Path mode at one worker count, with the base version and with this checkout."""

    workers: int
    base_s: float
    head_s: float


@dataclass(frozen=True)
class Version:
    """An aclpubcheck to run path mode with."""

    python: str
    pythonpath: str | None  # a checkout to import it from, or None for the interpreter's own


@dataclass(frozen=True)
class NameCheckCost:
    """One PDFNameCheck() (the rebiber database), which Formatter() used to build eagerly."""

    seconds: float
    peak_rss_mb: float | None  # peak memory of the interpreter that built it


@dataclass(frozen=True)
class Measurements:
    label: str
    corpus: Corpus
    records: int
    imports: list[tuple[float, float]]  # (import s, whole process s) per fresh interpreter
    formatter_s: float  # median Formatter() construction, lazy name check
    name_check: NameCheckCost | None  # None with --skip-eager
    timings: list[PaperTiming]
    runs: list[BatchRun]
    base_label: str | None  # None without --base-python
    path_mode: list[PathModeRun]
    reports_match_base: bool | None


_started = time.monotonic()


def module_location() -> str:
    """Where aclpubcheck was imported from, relative to the checkout so no home path is recorded."""
    module = Path(sys.modules["aclpubcheck"].__file__ or "").resolve().parent
    return str(module.relative_to(REPO)) if module.is_relative_to(REPO) else str(module)


def note(message: str) -> None:
    print(f"[{time.monotonic() - _started:7.1f}s] {message}", file=sys.stderr, flush=True)


def generate_corpus(root: Path, count: int) -> Corpus:
    papers = root / "papers"
    papers.mkdir(parents=True)
    names = list(KINDS)
    entries = []
    kinds = {}
    for number in range(1, count + 1):
        kind = names[(number - 1) % len(names)]
        write_pages(papers / f"{number}.pdf", list(KINDS[kind].pages))
        kinds[str(number)] = kind
        entries.append(
            {
                "id": number,
                "file": f"{number}.pdf",
                "title": f"Synthetic {kind} paper {number}",
                "attributes": {"paper_type": "long"},
                "authors": [{"name": "Ada Lovelace", "email": "ada@example.org"}],
            }
        )
    manifest = root / "papers.yml"
    manifest.write_text(yaml.safe_dump(entries), encoding="utf8")
    return Corpus("generated", manifest, papers, kinds)


def directory_corpus(root: Path, pdf_dir: Path, paper_type: str) -> Corpus:
    """A papers.yml listing every PDF directly under pdf_dir, all of one paper type."""
    entries = [
        {
            "id": number,
            "file": path.name,
            "title": path.stem,
            "attributes": {"paper_type": paper_type},
        }
        for number, path in enumerate(sorted(pdf_dir.glob("*.pdf")), 1)
    ]
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / "papers.yml"
    manifest.write_text(yaml.safe_dump(entries), encoding="utf8")
    return Corpus(str(pdf_dir), manifest, pdf_dir, {})


def import_times(repeat: int) -> list[tuple[float, float]]:
    """Import and whole-process seconds per fresh interpreter: what each spawned worker pays."""
    times = []
    for _ in range(repeat):
        started = time.perf_counter()
        probe = subprocess.run(
            [sys.executable, "-c", IMPORT_PROBE],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=True,
        )
        times.append((float(probe.stdout), time.perf_counter() - started))
    return times


def formatter_construction(repeat: int) -> float:
    """Median seconds for one Formatter(), which no longer builds the name-check database."""
    times = []
    for _ in range(repeat):
        started = time.perf_counter()
        Formatter()
        times.append(time.perf_counter() - started)
    return statistics.median(times)


def name_check_construction() -> NameCheckCost:
    probe = subprocess.run(
        [sys.executable, "-c", NAME_CHECK_PROBE],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    )
    return NameCheckCost(**json.loads(probe.stdout))


def page_count(path: Path) -> int:
    with pdfplumber.open(path) as pdf:
        return len(pdf.pages)


def semantic(results: Sequence[PaperResult]) -> dict[int, Semantic]:
    return {r.record.index: (r.status, r.errors, r.warnings, r.message) for r in results}


def differing(expected: dict[int, Semantic], actual: dict[int, Semantic]) -> list[int]:
    return sorted(i for i in expected.keys() | actual.keys() if expected.get(i) != actual.get(i))


def time_checks(
    records: Sequence[PaperRecord], corpus: Corpus, report_root: Path
) -> tuple[list[PaperTiming], dict[int, Semantic]]:
    """run_check on each paper in this process, one after another."""
    timings = []
    outcomes = {}
    for number, record in enumerate(records, 1):
        assert record.paper_type is not None  # only records without problems are timed
        path = corpus.papers_dir / record.file
        started = time.perf_counter()
        outcome = run_check(CheckJob(path, record.paper_type, report_root / str(record.index)))
        seconds = time.perf_counter() - started
        outcomes[record.index] = (outcome.status, outcome.errors, outcome.warnings, outcome.message)
        pages = None if outcome.status is Status.CHECK_ERROR else page_count(path)
        kind = corpus.kinds.get(record.paper_id, "pdf")
        timings.append(PaperTiming(record.paper_id, kind, pages, seconds, outcome.status.value))
        note(f"[{number}/{len(records)}] {record.paper_id} {outcome.status.value} {seconds:.3f}s")
    return timings, outcomes


def time_batch(
    records: Sequence[PaperRecord],
    provider: LocalPdfProvider,
    report_root: Path,
    workers: int,
    repeat: int,
) -> tuple[BatchRun, dict[int, Semantic]]:
    """One run_batch over every record, as the aclpubcheck command runs it."""
    first: list[float] = []

    def on_event(event: Event) -> None:
        result = event.result if isinstance(event, PaperChanged) else None
        if result and result.status.terminal and result.report_dir is not None and not first:
            first.append(time.perf_counter() - started)

    options = RunOptions(report_root=report_root, num_workers=workers)
    started = time.perf_counter()
    results = asyncio.run(run_batch(records, provider, options, fan_out(ConsoleSink(), on_event)))
    run = BatchRun(
        workers=workers,
        repeat=repeat,
        wall_s=time.perf_counter() - started,
        checked=sum(1 for r in results if r.report_dir is not None),
        first_result_s=first[0] if first else None,
        busy_s=sum(r.duration or 0.0 for r in results),
    )
    return run, semantic(results)


def time_path_mode(
    version: Version, records: Sequence[PaperRecord], papers_dir: Path, out: Path, workers: int
) -> float:
    """Check every paper as a chair would without batch mode: one run per paper type."""
    env = {name: value for name, value in os.environ.items() if name != "PYTHONPATH"}
    if version.pythonpath is not None:
        env["PYTHONPATH"] = version.pythonpath
    out.mkdir(parents=True)
    started = time.perf_counter()
    for paper_type in sorted({r.paper_type for r in records if r.paper_type}):
        files = [
            str((papers_dir / r.file).resolve()) for r in records if r.paper_type == paper_type
        ]
        command = [version.python, "-m", "aclpubcheck", "-p", paper_type]
        command += ["--num_workers", str(workers), "-o", str(out / paper_type), *files]
        # cwd is out, so no aclpubcheck/ directory in the working directory shadows the version
        done = subprocess.run(
            command, cwd=out, env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True
        )
        if done.returncode != 0:
            raise SystemExit(f"{version.python} -m aclpubcheck failed:\n{done.stderr[-4000:]}")
        # path mode skips files it does not take for PDFs and still exits 0
        written = len(list((out / paper_type).glob("errors-*.json")))
        if written != len(set(files)):
            raise SystemExit(
                f"{version.python} -m aclpubcheck wrote {written} reports for "
                f"{len(set(files))} {paper_type} papers"
            )
    return time.perf_counter() - started


def json_reports(directory: Path) -> dict[str, object]:
    return {
        str(path.relative_to(directory)): json.loads(path.read_text(encoding="utf8"))
        for path in sorted(directory.rglob("errors-*.json"))
    }


def compare_with_base(
    args: argparse.Namespace, records: Sequence[PaperRecord], papers_dir: Path, work: Path
) -> tuple[list[PathModeRun], bool]:
    """Path mode with the base version and with this checkout, on the same machine."""
    base = Version(args.base_python, None)
    head = Version(sys.executable, str(REPO))
    runs = []
    same = True
    for workers in args.workers:
        note(f"path mode with {workers} workers: {args.base_label}, then this checkout")
        base_out, head_out = work / f"base-{workers}", work / f"head-{workers}"
        base_s = time_path_mode(base, records, papers_dir, base_out, workers)
        head_s = time_path_mode(head, records, papers_dir, head_out, workers)
        runs.append(PathModeRun(workers, base_s, head_s))
        same = same and json_reports(base_out) == json_reports(head_out)
    return runs, same


def git(*args: str) -> str:
    out = subprocess.run(
        ["git", "-C", str(REPO), *args], capture_output=True, text=True, check=False
    )
    return out.stdout.strip() if out.returncode == 0 else ""


def cpu_model() -> str:
    if sys.platform == "darwin":
        out = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            check=False,
        )
        return out.stdout.strip() or platform.processor()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf8").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor()


def package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not installed"


def load_average() -> tuple[float, float, float] | None:
    """Other work on the machine skews every timing here, so it is recorded with them."""
    return os.getloadavg() if hasattr(os, "getloadavg") else None


def machine_info(load_before: tuple[float, float, float] | None) -> dict[str, object]:
    affinity = getattr(os, "sched_getaffinity", None)  # Linux: the CPUs this process may use
    return {
        "load_average_before": load_before,
        "load_average_after": load_average(),
        "platform": platform.platform(),
        "arch": platform.machine(),
        "cpu": cpu_model(),
        "cpu_count": os.cpu_count(),
        "usable_cpus": len(affinity(0)) if affinity else os.cpu_count(),
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
    }


def best_runs(runs: Sequence[BatchRun]) -> list[BatchRun]:
    """The fastest repeat for each worker count, in the order measured."""
    best: dict[int, BatchRun] = {}
    for run in runs:
        if run.workers not in best or run.wall_s < best[run.workers].wall_s:
            best[run.workers] = run
    return list(best.values())


def by_kind(timings: Sequence[PaperTiming]) -> dict[str, Stats]:
    kinds = dict.fromkeys(t.kind for t in timings)
    return {kind: Stats.of([t.seconds for t in timings if t.kind == kind]) for kind in kinds}


def table(m: Measurements, machine: dict[str, object]) -> str:
    imports = Stats.of([i for i, _ in m.imports])
    processes = Stats.of([p for _, p in m.imports])
    per_paper = Stats.of([t.seconds for t in m.timings])
    pages = sum(t.pages or 0 for t in m.timings)
    lines = [
        (
            f"aclpubcheck benchmark {m.label!r} on {machine['cpu']} "
            f"({machine['cpu_count']} CPUs), Python {machine['python']}"
        ),
        f"corpus: {m.records} papers, {len(m.timings)} checked, {pages} pages ({m.corpus.source})",
        "",
        (
            f"{'import aclpubcheck.formatchecker':36} {imports.median:9.3f} s   median of "
            f"{len(m.imports)} fresh interpreters; whole process {processes.median:.3f} s"
        ),
        f"{'Formatter() now (lazy name check)':36} {m.formatter_s * 1e6:9.1f} us",
    ]
    if m.name_check is not None:
        peak = m.name_check.peak_rss_mb
        lines.append(
            f"{'PDFNameCheck() (eager construction)':36} {m.name_check.seconds:9.3f} s   "
            f"{m.name_check.seconds / per_paper.median:.0f}x the median check, paid by every "
            "checked paper while Formatter() built it eagerly"
            + (f"; peak memory {peak:.0f} MB" if peak is not None else "")
        )
    lines.append(
        f"{'run_check per paper, in-process':36} {per_paper.median:9.3f} s   median "
        f"(min {per_paper.min:.3f}, max {per_paper.max:.3f})"
    )
    for kind, stats in by_kind(m.timings).items():
        count = sum(1 for t in m.timings if t.kind == kind)
        lines.append(
            f"  {kind:12} {count:4} papers {stats.median:13.3f} s   "
            f"(min {stats.min:.3f}, max {stats.max:.3f})"
        )
    runs = best_runs(m.runs)
    lines += [
        "",
        (
            f"{'workers':>7} {'wall s':>8} {'papers/s':>9} {'speedup':>8} "
            f"{'first result s':>15} {'concurrency':>12}"
        ),
    ]
    for run in runs:
        first = f"{run.first_result_s:.3f}" if run.first_result_s is not None else "-"
        lines.append(
            f"{run.workers:>7} {run.wall_s:>8.3f} {run.papers_per_s:>9.2f} "
            f"{runs[0].wall_s / run.wall_s:>7.2f}x {first:>15} {run.busy_s / run.wall_s:>12.2f}"
        )
    if m.base_label is not None:
        base = f"{m.base_label} s"
        lines += [
            "",
            "path mode, one `aclpubcheck -p TYPE FILES...` run per paper type",
            f"{'workers':>7} {base:>16} {'this checkout s':>16} {'speedup':>8}",
        ]
        for path_run in m.path_mode:
            lines.append(
                f"{path_run.workers:>7} {path_run.base_s:>16.1f} {path_run.head_s:>16.1f} "
                f"{path_run.base_s / path_run.head_s:>7.2f}x"
            )
        lines.append(
            f"reports identical to {m.base_label}: {'yes' if m.reports_match_base else 'NO'}"
        )
    return "\n".join(lines)


def report(m: Measurements, machine: dict[str, object], argv: Sequence[str]) -> dict[str, object]:
    return {
        "label": m.label,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "argv": list(argv),
        "machine": machine,
        "packages": {name: package_version(name) for name in PACKAGES},
        "aclpubcheck": {
            "commit": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
            "module": module_location(),
        },
        "corpus": {
            "source": m.corpus.source,
            "papers": m.records,
            "checked": len(m.timings),
            "pages": sum(t.pages or 0 for t in m.timings),
        },
        "import_formatchecker": [{"import_s": i, "process_s": p} for i, p in m.imports],
        "formatter_construction": {
            "lazy_formatter_s": m.formatter_s,
            "lazy_formatter_repeat": FORMATTER_REPEAT,
            "eager_name_check": asdict(m.name_check) if m.name_check else None,
        },
        "run_check": {
            "per_paper": asdict(Stats.of([t.seconds for t in m.timings])),
            "by_kind": {kind: asdict(stats) for kind, stats in by_kind(m.timings).items()},
            "papers": [asdict(t) for t in m.timings],
        },
        "run_batch": [
            {
                **asdict(run),
                "papers_per_s": run.papers_per_s,
                "concurrency": run.busy_s / run.wall_s,
            }
            for run in m.runs
        ],
        # the run fails before writing this file when worker counts disagree
        "results_identical_across_workers": True,
        "path_mode": None
        if m.base_label is None
        else {
            "base": m.base_label,
            "runs": [{**asdict(run), "speedup": run.base_s / run.head_s} for run in m.path_mode],
            "reports_match_base": m.reports_match_base,
        },
    }


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-") or "unnamed"


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument(
        "--label",
        default="local",
        help="names the results file, after the machine and corpus (default: local)",
    )
    parser.add_argument(
        "--papers", type=int, default=24, help="size of the generated corpus (default: 24)"
    )
    parser.add_argument(
        "--papers-yml", type=Path, help="measure the papers of an aclpub2 papers.yml"
    )
    parser.add_argument(
        "--pdf-dir",
        type=Path,
        help="the PDFs of --papers-yml (default: papers/ next to it); alone, every *.pdf in it",
    )
    parser.add_argument(
        "-p", "--paper-type", default="long", help="paper type for --pdf-dir without --papers-yml"
    )
    parser.add_argument(
        "--workers",
        default="1,2,4",
        help="comma-separated worker counts for run_batch and path mode (default: 1,2,4)",
    )
    parser.add_argument("--repeat", type=int, default=1, help="run_batch runs per worker count")
    parser.add_argument(
        "--import-repeat", type=int, default=3, help="fresh interpreters timing the import"
    )
    parser.add_argument(
        "--skip-eager",
        action="store_true",
        help="skip building PDFNameCheck() (seconds and ~1 GB), the per-paper cost while "
        "Formatter() built it eagerly",
    )
    parser.add_argument(
        "--base-python",
        help="an interpreter with another aclpubcheck installed, such as main; path mode is "
        "timed with it and with this checkout",
    )
    parser.add_argument(
        "--base-label", default="base", help="what to call the --base-python version"
    )
    parser.add_argument(
        "--results-dir", type=Path, default=Path(__file__).resolve().parent / "results"
    )
    args = parser.parse_args(argv)
    args.workers = list(dict.fromkeys(int(w) for w in args.workers.split(",")))
    if args.base_python is not None and os.sep in args.base_python:
        # path mode runs in another directory, and a venv's python must stay a symlink
        args.base_python = os.path.abspath(args.base_python)
    if min(args.workers) < 1 or min(args.repeat, args.papers, args.import_repeat) < 1:
        parser.error("--workers, --repeat, --papers and --import-repeat must be at least 1")
    return args


def load_corpus(args: argparse.Namespace, work: Path) -> Corpus:
    if args.papers_yml is not None:
        pdf_dir = args.pdf_dir or args.papers_yml.parent / "papers"
        return Corpus(str(args.papers_yml), args.papers_yml, pdf_dir, {})
    if args.pdf_dir is not None:
        return directory_corpus(work / "corpus", args.pdf_dir, args.paper_type)
    return generate_corpus(work / "corpus", args.papers)


def measure(args: argparse.Namespace, work: Path) -> Measurements:
    corpus = load_corpus(args, work)
    records = load_papers_yml(corpus.manifest)
    checkable = [r for r in records if not r.problems and (corpus.papers_dir / r.file).is_file()]
    if not checkable:
        raise SystemExit(f"no checkable papers in {corpus.source}")
    note(f"corpus: {len(records)} papers, {len(checkable)} checkable ({corpus.source})")
    if args.base_python is not None:  # fail now rather than after every other measurement
        probe = subprocess.run(
            [args.base_python, "-c", "import aclpubcheck"], cwd=work, capture_output=True, text=True
        )
        if probe.returncode != 0:
            raise SystemExit(f"--base-python cannot import aclpubcheck:\n{probe.stderr[-4000:]}")

    note("timing the import of aclpubcheck.formatchecker in fresh interpreters")
    imports = import_times(args.import_repeat)
    formatter_s = formatter_construction(FORMATTER_REPEAT)
    name_check = None
    if not args.skip_eager:
        note("timing one PDFNameCheck(), which Formatter() used to build for every paper")
        name_check = name_check_construction()

    note("timing run_check per paper, in this process")
    timings, in_process = time_checks(checkable, corpus, work / "in-process")
    for record in checkable:
        kind = corpus.kinds.get(record.paper_id)
        if kind and in_process[record.index][0] is not KINDS[kind].expected:
            raise SystemExit(
                f"synthetic {kind} paper {record.paper_id} ended "
                f"{in_process[record.index][0].value}, expected {KINDS[kind].expected.value}"
            )

    provider = LocalPdfProvider(corpus.papers_dir)
    runs = []
    baseline: dict[int, Semantic] = {}
    for workers in args.workers:
        for repeat in range(1, args.repeat + 1):
            note(f"run_batch with {workers} workers (run {repeat}/{args.repeat})")
            run, results = time_batch(
                records, provider, work / f"batch-{workers}-{repeat}", workers, repeat
            )
            runs.append(run)
            baseline = baseline or results
            # scheduling must not change any result: fail before recording numbers
            mismatch = differing(baseline, results)
            if mismatch:
                raise SystemExit(
                    f"{workers} workers disagree with {args.workers[0]} on indexes {mismatch}"
                )
            mismatch = differing({i: results[i] for i in in_process}, in_process)
            if mismatch:
                raise SystemExit(
                    f"{workers} workers disagree with the in-process check on indexes {mismatch}"
                )
    path_mode: list[PathModeRun] = []
    reports_match_base = None
    if args.base_python is not None:
        # path mode in either version stops at a PDF it cannot open, where batch mode records
        # check_error, so those papers are left out of the comparison
        path_records = [r for r in checkable if in_process[r.index][0] is not Status.CHECK_ERROR]
        path_mode, reports_match_base = compare_with_base(
            args, path_records, corpus.papers_dir, work / "path-mode"
        )
    return Measurements(
        label=args.label,
        corpus=corpus,
        records=len(records),
        imports=imports,
        formatter_s=formatter_s,
        name_check=name_check,
        timings=timings,
        runs=runs,
        base_label=args.base_label if args.base_python is not None else None,
        path_mode=path_mode,
        reports_match_base=reports_match_base,
    )


def main(argv: Sequence[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    args = parse_args(argv)
    load_before = load_average()
    with tempfile.TemporaryDirectory(prefix="aclpubcheck-bench-") as directory:
        measurements = measure(args, Path(directory))
    machine = machine_info(load_before)
    print(table(measurements, machine))
    path = args.results_dir / f"{slug(args.label)}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    data = report(measurements, machine, argv)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf8")
    note(f"wrote {path}")
    return 0


if __name__ == "__main__":  # spawned check workers re-import this file as __mp_main__
    sys.exit(main())
