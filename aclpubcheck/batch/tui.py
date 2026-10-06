"""Full-screen view of a batch run: totals, per-paper status, a problem filter and details."""

from __future__ import annotations

import asyncio
import os
import signal
import time
from collections.abc import Sequence
from contextlib import suppress
from multiprocessing import resource_tracker
from typing import ClassVar

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import VerticalScroll
from textual.message import Message
from textual.widgets import DataTable, Footer, Static
from textual.worker import Worker, WorkerState

from .check import LOG_NAME
from .console import status_counts
from .model import (
    Event,
    Job,
    PaperChanged,
    PaperResult,
    RunFinished,
    RunStarted,
    StageChanged,
    Status,
    printable,
)

COLUMNS = ("#", "paper", "status", "type", "duration", "title")
_STYLES = {Status.PASSED: "green", Status.WARNINGS: "yellow"}  # problems are red, in-flight cyan


def row_cells(result: PaperResult) -> tuple[Text, ...]:
    # Text, not str: DataTable parses str cells as markup, and titles may contain "[...]"
    record, status = result.record, result.status
    style = "bold red" if status.problem else _STYLES.get(status, "cyan")
    return (
        Text(str(record.index + 1)),
        Text(printable(record.paper_id)),
        Text(status.value, style=style),
        Text(record.paper_type or ""),
        Text("" if result.duration is None else f"{result.duration:.1f}s"),
        Text(printable(record.title)),
    )


def describe(result: PaperResult) -> str:
    """The detail panel for one paper: why it ended as it did and where its reports are."""
    record, pdf, report_dir = result.record, result.pdf, result.report_dir
    fields: list[tuple[str, Sequence[str]]] = [
        ("status", [result.status.value]),
        ("message", [result.message]),
        ("errors", [f"{f.category}: {f.message}" for f in result.errors]),
        ("warnings", [f"{f.category}: {f.message}" for f in result.warnings]),
        ("input notes", record.notes),
        ("report dir", [str(report_dir or "")]),
        ("report files", [", ".join(result.report_files)]),
        ("check log", [str(report_dir / LOG_NAME)] if report_dir else []),
        ("pdf", [str(pdf.path), f"sha256 {pdf.sha256}"] if pdf else []),
        ("source", [f"{record.source} {record.source_id}".strip()]),
    ]
    # every value is author- or PDF-controlled text, so control characters are neutralised
    lines = [printable(f"#{record.index + 1} {record.paper_id}  {record.title}").rstrip()]
    for label, values in fields:
        for number, value in enumerate(v for v in values if v):
            lines.append(f"{label if number == 0 else '':<14}{printable(value)}")
    return "\n".join(lines)


class BatchApp(App[None]):
    """Runs a batch job as a worker and shows its events; quitting cancels the job and waits."""

    ENABLE_COMMAND_PALETTE = False
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("p", "toggle_problems", "Problems only"),
        Binding("q", "quit", "Quit"),
        # priority: the screen's copy binding would otherwise take ctrl+c while text is selected
        Binding("ctrl+c", "quit", "Quit", show=False, priority=True),
    ]
    CSS = """
    #header { height: auto; padding: 0 1; background: $boost; }
    #papers { height: 1fr; }
    #detail-pane { height: 40%; padding: 0 1; border-top: solid $accent; }
    """

    class Published(Message):
        """One job event, queued so the job never waits for the UI."""

        def __init__(self, event: Event) -> None:
            super().__init__()
            self.event = event

    def __init__(self, job: Job, title: str) -> None:
        super().__init__()
        self.job = job
        self.title = title
        self.results: tuple[PaperResult, ...] | None = None  # what the job returned
        self.error: BaseException | None = None  # what the job raised
        self.problems_only = False
        self._cancel = asyncio.Event()
        self._papers: dict[int, PaperResult] = {}  # latest result per record index, input order
        self._stage = "starting"
        self._finished: RunFinished | None = None
        self._quitting = False
        self._started = time.monotonic()
        self._ended: float | None = None

    def compose(self) -> ComposeResult:
        yield Static(id="header", markup=False)
        yield DataTable(id="papers", cursor_type="row", zebra_stripes=True)
        with VerticalScroll(id="detail-pane"):
            yield Static(id="detail", markup=False)
        yield Footer()

    def on_load(self) -> None:
        # Textual swaps sys.stderr for a capture whose fileno() is -1 once the terminal is
        # taken over, and multiprocessing's resource tracker passes sys.stderr's fd to the
        # process it starts, so the check pool could not start inside the app: start it now
        if os.name == "posix":
            resource_tracker.ensure_running()

    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        for label in COLUMNS:
            table.add_column(label, key=label)
        self._show_header()
        self.set_interval(1, self._show_header)
        # `kill` cancels like q, so the summary is finished and the terminal restored;
        # Ctrl-C inside the app is a key press, but SIGINT can still come from outside
        for number in (signal.SIGTERM, signal.SIGINT):
            with suppress(NotImplementedError):  # no loop signal handlers on Windows
                asyncio.get_running_loop().add_signal_handler(
                    number, lambda: self.call_later(self.action_quit)
                )
        # with eager tasks (Python 3.12+) the job starts publishing right here
        self.run_worker(self.job(self._publish, self._cancel), exit_on_error=False)

    def on_unmount(self) -> None:
        for number in (signal.SIGTERM, signal.SIGINT):
            with suppress(NotImplementedError):
                asyncio.get_running_loop().remove_signal_handler(number)

    def _publish(self, event: Event) -> None:
        # never touches widgets, so it cannot fail and starve the sinks after it of events
        self.post_message(self.Published(event))

    def on_batch_app_published(self, message: BatchApp.Published) -> None:
        event = message.event
        if isinstance(event, StageChanged):
            self._stage = f"{event.stage}: {event.detail}" if event.detail else event.stage
        elif isinstance(event, RunStarted):
            self._stage = "checking"
            self._papers = {result.record.index: result for result in event.results}
            self._fill_table()
        elif isinstance(event, PaperChanged):
            self._update_row(event.result)
        else:
            self._finished = event
        self._show_header()

    def on_worker_state_changed(self, message: Worker.StateChanged) -> None:
        worker: Worker[tuple[PaperResult, ...]] = message.worker
        if message.state is WorkerState.SUCCESS:
            self.results = worker.result
        elif message.state is WorkerState.ERROR:
            self.error = worker.error
        else:
            return
        self._ended = time.monotonic()
        self._show_header()
        if self._quitting or self.error is not None:
            self.exit()

    def on_data_table_row_highlighted(self) -> None:
        self._show_detail()

    def action_toggle_problems(self) -> None:
        self.problems_only = not self.problems_only
        self._fill_table()
        self._show_header()

    async def action_quit(self) -> None:
        """Exit once the job has returned; a running job is cancelled first."""
        if self.results is not None or self.error is not None:
            self.exit()
            return
        self._quitting = True
        self._cancel.set()
        self._show_header()

    def _visible(self, result: PaperResult) -> bool:
        return not self.problems_only or result.status.problem

    def _highlighted(self) -> int | None:
        table = self.query_one(DataTable)
        if not table.is_valid_coordinate(table.cursor_coordinate):
            return None
        key = table.coordinate_to_cell_key(table.cursor_coordinate).row_key.value
        return None if key is None else int(key)

    def _fill_table(self) -> None:
        table = self.query_one(DataTable)
        keep = self._highlighted()
        table.clear()
        for index, result in self._papers.items():
            if self._visible(result):
                table.add_row(*row_cells(result), key=str(index))
        if keep is not None and str(keep) in table.rows:
            table.move_cursor(row=table.get_row_index(str(keep)))
        self._show_detail()

    def _update_row(self, result: PaperResult) -> None:
        index = result.record.index
        self._papers[index] = result
        table = self.query_one(DataTable)
        key = str(index)
        shown = key in table.rows
        if shown and self._visible(result):
            for column, cell in zip(COLUMNS, row_cells(result)):
                table.update_cell(key, column, cell, update_width=True)
        elif shown or self._visible(result):
            self._fill_table()  # rows can only be appended, so joining the filtered view rebuilds it
        if index == self._highlighted():
            self._show_detail()

    def _show_detail(self) -> None:
        index = self._highlighted()
        if index is not None:
            text = describe(self._papers[index])
        else:
            text = "no problem papers" if self.problems_only and self._papers else ""
        self.query_one("#detail", Static).update(text)

    def _show_header(self) -> None:
        papers = tuple(self._papers.values())
        if self._finished is not None:
            stage = "cancelled" if self._finished.cancelled else "finished"
        elif self._quitting:
            stage = "cancelling…"
        else:
            stage = self._stage
        elapsed = int((self._ended or time.monotonic()) - self._started)
        parts = [
            stage,
            f"{elapsed // 60}:{elapsed % 60:02d}",
            f"{sum(result.status.terminal for result in papers)}/{len(papers)} done",
            status_counts(papers),
            "problems only" if self.problems_only else "",
        ]
        self.query_one("#header", Static).update(
            f"{self.title}\n" + " | ".join(part for part in parts if part)
        )


def run_tui(job: Job, *, title: str) -> tuple[PaperResult, ...]:
    """Run `job` under the full-screen view; return its results once the user quits.

    The terminal is restored before anything the job raised is re-raised.
    """
    app = BatchApp(job, title)
    app.run()
    if app.error is not None:
        raise app.error
    if app.results is None:
        raise RuntimeError("the batch view closed before the run finished")
    return app.results
