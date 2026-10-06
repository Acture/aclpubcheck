from __future__ import annotations

import asyncio
import importlib.util
import os
import signal
import sys
import tempfile
import time
import unittest
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path

from batch_fixtures import EXPECTED, write_sample

from aclpubcheck.batch.manifest import LocalPdfProvider, load_papers_yml
from aclpubcheck.batch.model import (
    EventSink,
    FetchedPdf,
    Finding,
    PaperChanged,
    PaperRecord,
    PaperResult,
    RunFinished,
    RunStarted,
    StageChanged,
    Status,
)
from aclpubcheck.batch.runner import RunOptions, run_batch

HAS_TEXTUAL = importlib.util.find_spec("textual") is not None
if HAS_TEXTUAL:
    from textual.pilot import Pilot
    from textual.widgets import DataTable, Static

    from aclpubcheck.batch.tui import BatchApp, describe, row_cells

RECORDS = tuple(
    PaperRecord(
        index=index,
        paper_id=paper_id,
        title=title,
        paper_type="long",
        source="papers.yml",
        source_id=paper_id,
    )
    for index, (paper_id, title) in enumerate(
        [("1", "Clean"), ("2", "Fonts [draft]"), ("3", "Overflow"), ("4", "Hedged"), ("5", "Late")]
    )
)
REPORT_DIR = Path("/out/reports/3-abababababab")
FINAL = (
    PaperResult(RECORDS[0], Status.PASSED, duration=1.5),
    PaperResult(RECORDS[1], Status.CHECK_ERROR, message="ValueError: no text", duration=0.5),
    PaperResult(
        RECORDS[2],
        Status.VIOLATIONS,
        pdf=FetchedPdf(Path("/papers/3.pdf"), "ab" * 32),
        errors=(Finding("Margin", "Text in the right margin on page 1"),),
        warnings=(Finding("Bibliography", "Couldn't find any references."),),
        report_dir=REPORT_DIR,
        report_files=("check.log", "errors-3.json"),
        duration=2.0,
    ),
    PaperResult(RECORDS[3], Status.WARNINGS, warnings=(Finding("Bibliography", "Too many arXiv"),)),
)  # RECORDS[4] never finishes, so it ends cancelled


class FakeJob:
    """Publishes `before`, waits for `gate`, publishes `after` unless cancelled, then ends
    every unfinished paper as cancelled and publishes RunFinished, like run_batch."""

    def __init__(self, before: Sequence[PaperResult], after: Sequence[PaperResult] = ()) -> None:
        self.before, self.after = before, after
        self.gate = asyncio.Event()
        self.cancel: asyncio.Event | None = None
        self.returned: tuple[PaperResult, ...] | None = None

    async def __call__(self, sink: EventSink, cancel: asyncio.Event) -> tuple[PaperResult, ...]:
        self.cancel = cancel
        latest = {record.index: PaperResult(record) for record in RECORDS}

        def publish(result: PaperResult) -> None:
            latest[result.record.index] = result
            sink(PaperChanged(result))

        sink(StageChanged("loading", "papers.yml"))
        await asyncio.sleep(0.01)
        sink(RunStarted(tuple(latest.values())))
        for result in self.before:
            await asyncio.sleep(0.01)
            publish(result)
        await self.gate.wait()
        for result in () if cancel.is_set() else self.after:
            await asyncio.sleep(0.01)
            publish(result)
        for result in list(latest.values()):
            if not result.status.terminal:
                publish(replace(result, status=Status.CANCELLED))
        self.returned = tuple(latest.values())
        sink(RunFinished(self.returned, cancel.is_set()))
        return self.returned


async def until(predicate: Callable[[], bool], timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("timed out waiting for the app")
        await asyncio.sleep(0.01)


def rows(app: BatchApp) -> list[tuple[str, str]]:
    """(paper id, status) of every shown row, top to bottom."""
    table = app.query_one(DataTable)
    shown = (table.get_row_at(i) for i in range(table.row_count))
    return [(str(row[1]), str(row[2])) for row in shown]


def text(app: BatchApp, widget_id: str) -> str:
    return str(app.query_one(f"#{widget_id}", Static).content)


@unittest.skipUnless(HAS_TEXTUAL, "needs textual: pip install textual")
class BatchAppTest(unittest.IsolatedAsyncioTestCase):
    async def test_rows_and_header_follow_the_events(self) -> None:
        job = FakeJob(
            before=[PaperResult(RECORDS[0], Status.CHECKING), FINAL[0], FINAL[1]],
            after=FINAL[2:],
        )
        app = BatchApp(job, "aclpubcheck: papers.yml")
        async with app.run_test() as pilot:
            running = [("1", "passed"), ("2", "check_error"), ("3", "queued"), ("4", "queued")]
            await until(lambda: rows(app)[:4] == running)
            header = text(app, "header")
            self.assertTrue(header.startswith("aclpubcheck: papers.yml\nchecking | "), header)
            self.assertIn("2/5 done | 3 queued, 1 passed, 1 check_error", header)

            job.gate.set()
            await until(lambda: app.results is not None)
            await pilot.pause()
            self.assertEqual(
                rows(app),
                [
                    ("1", "passed"),
                    ("2", "check_error"),
                    ("3", "violations"),
                    ("4", "warnings"),
                    ("5", "cancelled"),
                ],
            )
            table = app.query_one(DataTable)
            self.assertEqual(
                [str(c) for c in table.get_row("0")], ["1", "1", "passed", "long", "1.5s", "Clean"]
            )
            self.assertEqual(str(table.get_row("1")[-1]), "Fonts [draft]")  # not markup
            header = text(app, "header")
            self.assertIn("\nfinished | ", header)
            self.assertIn(
                "5/5 done | 1 passed, 1 warnings, 1 violations, 1 check_error, 1 cancelled",
                header,
            )

            await pilot.press("ctrl+c")  # the job has returned, so this exits at once
            await until(lambda: app.return_code is not None)
        self.assertIs(app.results, job.returned)

    async def test_problem_filter(self) -> None:
        job = FakeJob(
            before=[FINAL[0], PaperResult(RECORDS[1], Status.CHECKING), FINAL[2]],
            after=[FINAL[1], FINAL[3]],
        )
        app = BatchApp(job, "sample")
        async with app.run_test() as pilot:
            await until(lambda: ("3", "violations") in rows(app))
            await pilot.press("p")
            # in-flight and queued papers are not problems
            self.assertEqual(rows(app), [("3", "violations")])
            self.assertIn("problems only", text(app, "header"))

            job.gate.set()
            await until(lambda: app.results is not None)
            await pilot.pause()
            # a paper that becomes a problem joins the view in input order
            self.assertEqual(
                rows(app), [("2", "check_error"), ("3", "violations"), ("5", "cancelled")]
            )
            await pilot.press("p")
            self.assertEqual(
                [status for _, status in rows(app)],
                ["passed", "check_error", "violations", "warnings", "cancelled"],
            )
            self.assertNotIn("problems only", text(app, "header"))

    async def test_detail_panel_shows_findings_and_reports(self) -> None:
        job = FakeJob(before=FINAL)
        job.gate.set()
        app = BatchApp(job, "sample")
        async with app.run_test(size=(120, 40)) as pilot:
            await until(lambda: app.results is not None)
            await pilot.pause()
            self.assertIn("status        passed", text(app, "detail"))

            await pilot.press("down", "down")
            await pilot.pause()
            detail = text(app, "detail")
            for expected in (
                "status        violations",
                "Margin: Text in the right margin on page 1",
                "Bibliography: Couldn't find any references.",
                f"report dir    {REPORT_DIR}",
                "check.log, errors-3.json",
                f"check log     {REPORT_DIR / 'check.log'}",
                "/papers/3.pdf",
                f"sha256 {'ab' * 32}",
                "source        papers.yml 3",
            ):
                self.assertIn(expected, detail)

            # filtering keeps the highlighted paper
            await pilot.press("p")
            await pilot.pause()
            self.assertIn("status        violations", text(app, "detail"))
            await pilot.press("up")
            await pilot.pause()
            self.assertIn("message       ValueError: no text", text(app, "detail"))

    async def test_quit_cancels_and_waits_for_the_job(self) -> None:
        job = FakeJob(before=[FINAL[0]], after=[FINAL[2]])
        app = BatchApp(job, "sample")
        async with app.run_test() as pilot:
            await until(lambda: rows(app)[:1] == [("1", "passed")])
            await pilot.press("q")
            await pilot.pause()
            assert job.cancel is not None
            self.assertTrue(job.cancel.is_set())
            self.assertIn("cancelling…", text(app, "header"))
            await asyncio.sleep(0.1)
            self.assertIsNone(app.return_code)  # still waiting for the job

            job.gate.set()
            await until(lambda: app.return_code is not None)
        self.assertIs(app.results, job.returned)
        self.assertEqual(
            [result.status for result in app.results or ()],
            [Status.PASSED] + [Status.CANCELLED] * 4,
        )

    @unittest.skipIf(sys.platform == "win32", "needs POSIX signals")
    async def test_sigterm_cancels_like_quit(self) -> None:
        job = FakeJob(before=[FINAL[0]], after=[FINAL[2]])
        app = BatchApp(job, "sample")
        async with app.run_test():
            await until(lambda: rows(app)[:1] == [("1", "passed")])
            os.kill(os.getpid(), signal.SIGTERM)
            await until(lambda: job.cancel is not None and job.cancel.is_set())
            self.assertIsNone(app.return_code)  # still waiting for the job
            job.gate.set()
            await until(lambda: app.return_code is not None)
        self.assertIs(app.results, job.returned)

    async def test_job_error_is_kept_for_run_tui(self) -> None:
        error = ValueError("papers.yml: expected a list of papers")

        async def job(sink: EventSink, cancel: asyncio.Event) -> tuple[PaperResult, ...]:
            sink(StageChanged("loading", "papers.yml"))
            await asyncio.sleep(0.01)
            raise error

        app = BatchApp(job, "sample")
        async with app.run_test():
            await until(lambda: app.return_code is not None)
        self.assertIs(app.error, error)
        self.assertIsNone(app.results)


@unittest.skipUnless(HAS_TEXTUAL, "needs textual: pip install textual")
class EscapeTest(unittest.TestCase):
    def test_author_text_cannot_reach_the_terminal_raw(self) -> None:
        record = PaperRecord(
            index=0,
            paper_id="1",
            title="Evil \x1b]0;PWNED\x1b\\ \x1b]8;;https://evil.example/\x1b\\x",
        )
        result = PaperResult(
            record,
            status=Status.VIOLATIONS,
            errors=(Finding("Font", "Wrong font. The main font used is Evil\x00\x1b]0;X\x07"),),
        )
        for text in [cell.plain for cell in row_cells(result)] + [describe(result)]:
            self.assertNotIn("\x1b", text)
            self.assertNotIn("\x00", text)


@unittest.skipUnless(HAS_TEXTUAL, "needs textual: pip install textual")
class RealRunnerTest(unittest.TestCase):
    def test_sample_run_ends_every_row(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        records = load_papers_yml(write_sample(root))
        provider = LocalPdfProvider(root / "papers")
        options = RunOptions(report_root=root / "reports", num_workers=2)

        async def job(sink: EventSink, cancel: asyncio.Event) -> tuple[PaperResult, ...]:
            return await run_batch(records, provider, options, sink, cancel)

        expected = [(record.paper_id, EXPECTED[record.paper_id]) for record in records]
        app = BatchApp(job, "sample")

        async def review(pilot: Pilot[object]) -> None:
            await until(lambda: app.results is not None or app.error is not None, timeout=300)
            if app.error is not None:
                raise app.error
            await pilot.pause()
            self.assertEqual(rows(app), expected)
            self.assertIn("\nfinished | ", text(app, "header"))
            self.assertIn(f"{len(records)}/{len(records)} done", text(app, "header"))
            await pilot.press("q")

        # App.run as run_tui uses it (run_test differs): eager tasks on 3.12+, stderr captured
        app.run(headless=True, auto_pilot=review)
        self.assertEqual(
            [(result.record.paper_id, result.status.value) for result in app.results or ()],
            expected,
        )


if __name__ == "__main__":
    unittest.main()
