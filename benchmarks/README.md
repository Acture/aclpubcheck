# Benchmarks

These benchmarks measure where batch checking spends its time, and how a change compares with the version it is based on. A claim about speed needs a number in `results/` behind it.

The numbers of record come from CI. The `benchmark` job in `.github/workflows/tests.yml` runs on a GitHub-hosted `ubuntu-latest` runner, so all of its results are measured on about the same hardware. The pool behind that label mixes CPU models, so compare absolute times between runs only when their `machine.cpu` matches. On a pull request the job also installs the base branch and times it on the same runner, so that comparison does not depend on which runner picked up the job. Timings from a laptop depend on the machine and on whatever else it is running, so they are not committed.

`tests/test_performance.py` is the regression gate: it fails CI when batch reports differ from single-file runs, when a format check builds the name-check database, or when 4 workers stop being clearly faster than 1. This benchmark only records numbers; it fails when results differ between worker counts.

## What `bench_checks.py` measures

1. **Import time** of `aclpubcheck.formatchecker` in a fresh interpreter, plus the whole process. The check pool uses `spawn`, so every worker pays this once each time a pool starts.
2. **`Formatter()` construction**, now and as it used to be. `Formatter()` used to build a `PDFNameCheck` (the rebiber database) in its constructor, so every checked paper paid for one, in path mode and in batch mode alike. Now the database is built only when the reference check needs it, and batch workers build it at most once per process. The benchmark times `Formatter()` as it is now and one `PDFNameCheck()` in a fresh interpreter, with its peak memory.
3. **`run_check` per paper**, in-process and one paper at a time: the CPU-bound unit of work. It reports min, median and max, both overall and per kind of paper.
4. **`run_batch` per worker count**, through `LocalPdfProvider` and a `papers.yml`, the same path the `aclpubcheck` command takes. Each worker count gets:
   - wall time;
   - papers/s, i.e. checked papers divided by wall time;
   - speedup over the first worker count;
   - *first result*, the time to the first finished check: pool start-up plus that check;
   - *concurrency*, the summed per-paper fetch and check time divided by wall time.

   If any paper's status, errors, warnings or message differ between worker counts, or differ from the in-process check, the run stops and writes no results.
5. **Path mode against a base version**, with `--base-python`: plain `aclpubcheck -p TYPE FILES...`, one run per paper type, as a chair would check the papers without batch mode. It runs once with the base version's interpreter and once with this checkout, at each worker count, and records whether the two wrote the same JSON reports.

Each results file also records:

- the machine: CPU, CPU count, platform, Python, and the load average before and after the run;
- the versions of pdfplumber, pdfminer.six, pypdfium2, numpy and Pillow;
- the aclpubcheck commit and whether the checkout had uncommitted changes.

## Results

`bench_checks.py` writes `results/<label>.json`. The label names the machine and the corpus; CI uses `github-ubuntu-latest-sigdial`, the aclpub2 SIGDIAL example at commit 4fd082b (41 papers), the test data suggested in acl-org/aclpubcheck#87. Each CI run shows the table in its summary and uploads the JSON as the `benchmark` artifact. The file committed here is that artifact from a run on the commit it records; git history keeps the earlier ones.

### SIGDIAL on `ubuntu-latest`

`results/github-ubuntu-latest-sigdial.json` comes from the `benchmark` job of CI run 37486891443, on commit 9edb957 of acl-org/aclpubcheck#89, with main at 1e33bca timed on the same runner: AMD EPYC 7763, 4 CPUs, Python 3.12.3, load average 0.5 before the run.

| | main | this checkout |
| --- | ---: | ---: |
| path mode, 1 worker | 151.4 s | 52.5 s (2.88x) |
| path mode, 4 workers | 67.1 s | 23.7 s (2.83x) |
| batch mode, 1 worker | | 52.7 s |
| batch mode, 4 workers | | 23.9 s (2.20x over 1 worker) |

| measurement | result |
| --- | --- |
| `run_check` per paper, in-process | median 1.31 s (min 0.49, max 4.49) |
| `PDFNameCheck()`, which every `Formatter()` used to build | 2.32 s, peak memory 915 MB |

What this shows:

- **The reports are identical to main's** for all 41 papers, and batch reports are identical to path mode; `tests/test_performance.py` checks the latter in every CI run. The run ends 25 passed and 16 violations.
- **The eager name-check build was most of main's time.** 41 papers × 2.32 s is 95 s, close to the 98.9 s that path mode with one worker saves.
- **Four workers check 2.2x as fast as one on this 4-CPU runner.** Concurrency reaches 3.74, so all four checks run at once, but each takes longer than when it runs alone.
- **The example's paper types are not reliable.** 13 of its 25 `short` papers are 9 to 14 pages long and fail the page limit. These are long papers labelled `short` in the demo data, not checker errors, and the summary makes that visible at a glance.

## Running

Run from a checkout with the package installed (`uv pip install -e .` or `pip install -e .`). The script imports the checkout's `aclpubcheck` and takes its synthetic PDFs from `tests/pdf_fixtures.py`.

```bash
# generated corpus: N synthetic long papers, cycling clean / text in the margin / 11 pages
python benchmarks/bench_checks.py --label local-synthetic

# real papers: an aclpub2 papers.yml (PDFs in papers/ next to it unless --pdf-dir says otherwise)
python benchmarks/bench_checks.py --label local-sigdial --papers-yml path/to/papers.yml --workers 1,4

# a plain directory of PDFs, all checked as one paper type
python benchmarks/bench_checks.py --label local-folder --pdf-dir path/to/pdfs -p long

# compared with main, installed into its own environment
python benchmarks/bench_checks.py --label local-sigdial --papers-yml path/to/papers.yml \
    --base-python path/to/main-venv/bin/python --base-label main
```

Other options:

| option | effect |
| --- | --- |
| `--papers N` | size of the generated corpus (default 24) |
| `--workers 1,2,4` | worker counts to run `run_batch` and path mode with |
| `--repeat R` | `run_batch` runs per worker count; the table shows the fastest |
| `--skip-eager` | skip building `PDFNameCheck()` |
| `--base-python`, `--base-label` | time path mode with another installed version too, and what to call it |
| `--results-dir` | where the JSON goes |

The script prints a table to stdout and progress to stderr. Never commit a real venue's `summary.csv`, `input.json` or logs: they hold author names and emails.

To fetch the SIGDIAL example as CI does:

```bash
git clone --filter=blob:none --no-checkout https://github.com/rycolab/aclpub2.git aclpub2
git -C aclpub2 sparse-checkout set examples/sigdial
git -C aclpub2 checkout 4fd082b0ec502a764e211fdf868d51d839225891
# papers.yml is aclpub2/examples/sigdial/papers.yml, the PDFs are in papers/ next to it
```
