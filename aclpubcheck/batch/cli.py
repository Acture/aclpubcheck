"""Batch mode of the aclpubcheck command: --papers-yml or --openreview-venue."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import sys
import tempfile
from contextlib import suppress
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

from ..formatchecker import CheckConfig
from .console import ConsoleSink
from .manifest import LocalPdfProvider, ManifestError, load_papers_yml
from .model import (
    EventSink,
    Job,
    MissingDependency,
    PaperRecord,
    PaperResult,
    PdfProvider,
    StageChanged,
    Status,
    describe_error,
    fan_out,
)
from .runner import RunOptions, run_batch
from .summary import SummaryWriter, write_summary

if TYPE_CHECKING:
    from .openreview_source import OpenReviewClient

EXIT_NO_PAPERS = 1
EXIT_USAGE = 2
EXIT_CANCELLED = 130


# options that only make sense with --papers-yml or --openreview-venue
_BATCH_ONLY = (
    "papers_dir",
    "openreview_baseurl",
    "openreview_pdf_field",
    "download_concurrency",
    "summary",
    "check_references",
)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group(
        "batch checking",
        "Check every paper listed by a papers.yml or an OpenReview venue and write one summary row "
        "per paper. Paper types come from the input, so -p does not apply. -o/--output-dir and "
        "--temp-output-dir choose OUTPUT_DIR (default: aclpubcheck-batch).",
    )
    source = group.add_mutually_exclusive_group()
    source.add_argument("--papers-yml", type=Path, metavar="PATH", help="aclpub2 papers.yml")
    source.add_argument(
        "--openreview-venue",
        metavar="VENUE_ID",
        help="OpenReview venue id; its accepted papers are the notes whose venueid equals it. "
        "Needs openreview-py. Credentials come from OPENREVIEW_USERNAME/OPENREVIEW_PASSWORD "
        "(a two-factor code is asked for on the terminal) or OPENREVIEW_TOKEN; without them "
        "only public notes are visible.",
    )
    group.add_argument(
        "--papers-dir",
        type=Path,
        metavar="DIR",
        help="directory holding the PDFs named in papers.yml (default: papers/ next to it)",
    )
    group.add_argument("--openreview-baseurl", default="https://api2.openreview.net", metavar="URL")
    group.add_argument(
        "--openreview-pdf-field",
        default="pdf",
        metavar="FIELD",
        help="note field holding the camera-ready PDF (default: pdf)",
    )
    group.add_argument(
        "--download-concurrency",
        type=int,
        default=1,
        metavar="N",
        help="parallel OpenReview downloads (default: 1; raise only within the server's rate limits)",
    )
    group.add_argument(
        "--summary",
        type=Path,
        metavar="PATH",
        help="summary file, tab-separated if it ends in .tsv (default: OUTPUT_DIR/summary.csv)",
    )
    group.add_argument(
        "--check-references",
        action="store_true",
        help="also run the bibliography checks, which only produce warnings; unless "
        "--disable_name_check is given they include the citation name check, which uploads "
        "each PDF to ref.scholarcy.com",
    )


def validate_mode(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Exit with a usage error unless the arguments ask for exactly one kind of run:
    PDF paths, or one batch source without the single-file options."""
    if is_batch(args):
        if args.submission_paths:
            parser.error("PDF paths cannot be combined with --papers-yml or --openreview-venue")
        if args.paper_type != parser.get_default("paper_type"):
            parser.error(
                "-p/--paper_type does not apply to batch checks: each paper's type comes from its input"
            )
        return
    if not args.submission_paths:
        parser.error("give PDF files or directories, --papers-yml or --openreview-venue")
    given = [
        "--" + dest.replace("_", "-")
        for dest in _BATCH_ONLY
        if getattr(args, dest) != parser.get_default(dest)
    ]
    if given:
        parser.error(f"{', '.join(given)} only apply with --papers-yml or --openreview-venue")


def resolve_output_dir(args: argparse.Namespace) -> Path:
    """The batch OUTPUT_DIR: -o/--output-dir, a new temporary directory with
    --temp-output-dir, otherwise aclpubcheck-batch."""
    if args.temp_output_dir:
        return Path(tempfile.mkdtemp(prefix="aclpubcheck-"))
    return Path(args.output_dir) if args.output_dir else Path("aclpubcheck-batch")


def is_batch(args: argparse.Namespace) -> bool:
    return args.papers_yml is not None or args.openreview_venue is not None


class _LoadCancelled(Exception):
    pass


_T = TypeVar("_T")


async def _unless_cancelled(cancel: asyncio.Event, work: asyncio.Future[_T]) -> _T:
    stop = asyncio.ensure_future(cancel.wait())
    await asyncio.wait({work, stop}, return_when=asyncio.FIRST_COMPLETED)
    stop.cancel()
    if work.done():
        return work.result()
    work.cancel()
    raise _LoadCancelled


async def _load(
    args: argparse.Namespace, client: OpenReviewClient | None, sink: EventSink
) -> tuple[list[PaperRecord], PdfProvider]:
    if client is None:
        sink(StageChanged("loading", str(args.papers_yml)))
        papers_dir = args.papers_dir or args.papers_yml.parent / "papers"
        return load_papers_yml(args.papers_yml), LocalPdfProvider(papers_dir)
    from .openreview_source import OpenReviewPdfProvider, load_openreview

    records = await load_openreview(
        client, args.openreview_venue, sink, pdf_field=args.openreview_pdf_field
    )
    provider = OpenReviewPdfProvider(
        client,
        args.output_dir / "pdfs",
        pdf_field=args.openreview_pdf_field,
    )
    return records, provider


def _connect(baseurl: str) -> OpenReviewClient:
    from .openreview_source import connect

    return connect(baseurl)


def _unwritable(directory: Path) -> str:
    """Why files cannot be created in `directory`, checked without creating anything."""
    for existing in (directory, *directory.parents):
        if existing.exists():
            if not existing.is_dir():
                return f"{existing} is not a directory"
            if not os.access(existing, os.W_OK | os.X_OK):
                return f"{existing} is not writable"
            return ""
    return ""


def _credentials() -> str:
    if os.environ.get("OPENREVIEW_TOKEN"):
        return "OPENREVIEW_TOKEN"
    if os.environ.get("OPENREVIEW_USERNAME"):
        return os.environ["OPENREVIEW_USERNAME"]
    return "anonymous (only public notes are visible)"


def _write_snapshot(path: Path, args: argparse.Namespace, records: list[PaperRecord]) -> None:
    """The input exactly as loaded, to tell later runs and revisions apart."""
    path.parent.mkdir(parents=True, exist_ok=True)
    source = {
        "papers_yml": str(args.papers_yml or ""),
        "openreview_venue": args.openreview_venue or "",
        "openreview_baseurl": args.openreview_baseurl if args.openreview_venue else "",
    }
    snapshot = {
        "loaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": source,
        "records": [asdict(record) for record in records],
    }
    path.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False), encoding="utf8")


def make_job(
    args: argparse.Namespace, config: CheckConfig, summary: Path, client: OpenReviewClient | None
) -> Job:
    async def job(sink: EventSink, cancel: asyncio.Event) -> tuple[PaperResult, ...]:
        records, provider = await _unless_cancelled(
            cancel, asyncio.ensure_future(_load(args, client, sink))
        )
        _write_snapshot(args.output_dir / "input.json", args, records)
        if not records:
            write_summary(summary, ())
            sink(StageChanged("done", "no papers found; check the input and its permissions"))
            return ()
        options = RunOptions(
            report_root=args.output_dir / "reports",
            num_workers=args.num_workers,
            download_concurrency=args.download_concurrency,
            check=config,
            check_references=args.check_references,
        )
        return await run_batch(
            records, provider, options, fan_out(sink, SummaryWriter(summary)), cancel
        )

    return job


async def _run_plain(job: Job, sink: EventSink) -> tuple[PaperResult, ...]:
    cancel = asyncio.Event()
    loop = asyncio.get_running_loop()
    signals = (signal.SIGINT, signal.SIGTERM)

    def stop() -> None:
        if not cancel.is_set():
            print("cancelling: unfinished papers will be marked cancelled", file=sys.stderr)
            cancel.set()

    for number in signals:
        with suppress(NotImplementedError):  # no loop signal handlers on Windows
            loop.add_signal_handler(number, stop)
    try:
        return await job(sink, cancel)
    finally:
        for number in signals:
            with suppress(NotImplementedError):
                loop.remove_signal_handler(number)


def run(args: argparse.Namespace, config: CheckConfig) -> int:
    """Run a batch from parsed arguments; returns the process exit code."""
    if args.download_concurrency < 1 or args.num_workers < 1:
        print("--download-concurrency and --num_workers must be at least 1", file=sys.stderr)
        return EXIT_USAGE
    args.output_dir = resolve_output_dir(args)
    summary = args.summary or args.output_dir / "summary.csv"
    for directory in (args.output_dir, summary.parent):
        problem = _unwritable(directory)
        if problem:
            print(f"cannot write to {directory}: {problem}", file=sys.stderr)
            return EXIT_USAGE
    print(f"Saving reports to {args.output_dir}", file=sys.stderr)
    client = None
    if args.openreview_venue:
        # before any UI: a two-factor login may prompt on the terminal
        print(f"connecting to {args.openreview_baseurl} as {_credentials()}", file=sys.stderr)
        try:
            client = _connect(args.openreview_baseurl)
        except MissingDependency as error:
            print(error, file=sys.stderr)
            return EXIT_USAGE
        except KeyboardInterrupt:
            return EXIT_CANCELLED
        except Exception as error:  # noqa: BLE001 -- a login failure is the user's to fix
            print(f"cannot connect to OpenReview: {describe_error(error)}", file=sys.stderr)
            return EXIT_USAGE
    job = make_job(args, config, summary, client)
    try:
        results = asyncio.run(_run_plain(job, ConsoleSink()))
    except ManifestError as error:
        print(error, file=sys.stderr)
        return EXIT_USAGE
    except _LoadCancelled:
        print("cancelled while loading the papers", file=sys.stderr)
        return EXIT_CANCELLED
    print(f"summary: {summary}", file=sys.stderr)
    if not results:
        return EXIT_NO_PAPERS
    if any(result.status is Status.CANCELLED for result in results):
        return EXIT_CANCELLED
    return 0
