from __future__ import annotations

import asyncio
import csv
import hashlib
import io
import os
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
from collections.abc import Mapping
from contextlib import redirect_stderr
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import overload
from unittest import mock

from batch_fixtures import Recorder, semantic
from pdf_fixtures import MARGIN_TEXT, TEXT, write_pdf

from aclpubcheck import formatchecker
from aclpubcheck.batch.model import (
    Event,
    MissingDependency,
    PaperChanged,
    PaperResult,
    StageChanged,
    Status,
)
from aclpubcheck.batch.openreview_source import OpenReviewPdfProvider, connect, load_openreview
from aclpubcheck.batch.runner import RunOptions, run_batch
from aclpubcheck.batch.summary import COLUMNS, SummaryWriter

VENUE = "aclweb.org/ACL/2026/Conference"
BASEURL = "https://api2.openreview.net"
ADA = ("Ada Lovelace", "~Ada_Lovelace1")


def pdf_bytes(content: bytes) -> bytes:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "paper.pdf"
        write_pdf(path, content)
        return path.read_bytes()


CLEAN = pdf_bytes(TEXT)
MARGIN = pdf_bytes(MARGIN_TEXT)


@dataclass
class FakeNote:
    """openreview.api.Note as the SDK returns it: content values are {"value": ...}."""

    id: str
    number: int
    content: dict[str, dict[str, object]]
    mdate: int | None = None
    tmdate: int | None = None


@dataclass
class FakeProfile:
    id: str
    content: dict[str, object]


def confirmed_emails(profile: FakeProfile) -> list[str]:
    emails = profile.content.get("emailsConfirmed", [])
    return [str(email) for email in emails] if isinstance(emails, list) else []


class FakeOpenReviewException(Exception):
    """Shaped like openreview.OpenReviewException: the API's JSON error is the only argument."""


def note(
    number: int,
    *,
    venueid: str = VENUE,
    paper_type: str | None = "Long Paper",
    pdf: str | None = None,
    authors: list[tuple[str, str]] | None = None,
    unified: bool = False,
    mdate: int = 1_700_000_000_000,
) -> FakeNote:
    listed = [ADA] if authors is None else authors
    content: dict[str, dict[str, object]] = {
        "title": {"value": f"Paper {number}"},
        "venueid": {"value": venueid},
        "pdf": {"value": pdf if pdf is not None else f"/pdf/{number:040x}.pdf"},
    }
    if unified:
        content["authors"] = {"value": [{"fullname": n, "username": i} for n, i in listed]}
    else:
        content["authors"] = {"value": [n for n, _ in listed]}
        content["authorids"] = {"value": [i for _, i in listed]}
    if paper_type is not None:
        content["paper_type"] = {"value": paper_type}
    return FakeNote(f"note{number}", number, content, mdate=mdate, tmdate=mdate + 1)


def usernames(profile: FakeProfile) -> set[str]:
    names = profile.content.get("names", [])
    assert isinstance(names, list)
    return {profile.id, *(n["username"] for n in names if "username" in n)}


class FakeClient:
    """In-memory OpenReviewClient recording every call; downloads track how many overlap."""

    def __init__(
        self,
        notes: list[FakeNote],
        pdfs: dict[str, bytes | Exception] | None = None,
        profiles: tuple[FakeProfile, ...] = (),
        profile_errors: dict[str, Exception] | None = None,
        delay: float = 0.0,
    ) -> None:
        self.notes = notes
        self.pdfs = pdfs if pdfs is not None else {n.id: CLEAN for n in notes}
        self.profiles = profiles
        self.profile_errors = profile_errors or {}
        self.delay = delay
        self.gates: dict[str, threading.Event] = {}
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.max_in_flight = 0
        self._in_flight = 0
        self._lock = threading.Lock()

    def _record(self, name: str, **kwargs: object) -> None:
        with self._lock:
            self.calls.append((name, kwargs))

    def calls_to(self, name: str) -> list[dict[str, object]]:
        return [kwargs for called, kwargs in self.calls if called == name]

    def get_all_notes(self, *, content: Mapping[str, str]) -> list[FakeNote]:
        self._record("get_all_notes", content=content)
        return list(self.notes)  # unfiltered, so the loader's own venueid filter is exercised

    @overload
    def search_profiles(self, *, ids: list[str]) -> list[FakeProfile]: ...

    @overload
    def search_profiles(self, *, confirmedEmails: list[str]) -> dict[str, FakeProfile]: ...

    def search_profiles(
        self, *, confirmedEmails: list[str] | None = None, ids: list[str] | None = None
    ) -> list[FakeProfile] | dict[str, FakeProfile]:
        kind = "ids" if ids is not None else "confirmedEmails"
        self._record("search_profiles", **{kind: list(ids or confirmedEmails or [])})
        if kind in self.profile_errors:
            raise self.profile_errors[kind]
        if ids is not None:
            return [p for p in self.profiles if usernames(p) & set(ids)]
        wanted = {email.lower() for email in confirmedEmails or []}
        # keyed by the profile's confirmed emails, as the SDK does
        return {
            email: p
            for p in self.profiles
            for email in confirmed_emails(p)
            if email.lower() in wanted
        }

    def get_pdf(self, id: str) -> bytes:
        self._record("get_pdf", id=id)
        return self._serve(id)

    def get_attachment(self, field_name: str, id: str) -> bytes:
        self._record("get_attachment", field_name=field_name, id=id)
        return self._serve(id)

    def _serve(self, note_id: str) -> bytes:
        with self._lock:
            self._in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self._in_flight)
        try:
            gate = self.gates.get(note_id)
            if gate is not None and not gate.wait(timeout=30):
                raise TimeoutError(f"download of {note_id} was never released")
            time.sleep(self.delay)
            served = self.pdfs[note_id]
        finally:
            with self._lock:
                self._in_flight -= 1
        if isinstance(served, Exception):
            raise served
        return served


class OpenReviewTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    async def run_records(
        self,
        client: FakeClient,
        out: str = "out",
        concurrency: int = 1,
        pdf_field: str = "pdf",
        workers: int = 1,
    ) -> tuple[PaperResult, ...]:
        records = await load_openreview(client, VENUE, Recorder(), pdf_field=pdf_field)
        provider = OpenReviewPdfProvider(client, self.root / out / "pdfs", pdf_field=pdf_field)
        options = RunOptions(
            report_root=self.root / out / "reports",
            num_workers=workers,
            download_concurrency=concurrency,
        )
        return await run_batch(records, provider, options, Recorder())


class StatusSequenceTest(OpenReviewTestCase):
    async def test_a_downloaded_paper_waits_as_queued(self) -> None:
        client = FakeClient([note(1), note(2)])
        recorder = Recorder()
        records = await load_openreview(client, VENUE, Recorder())
        provider = OpenReviewPdfProvider(client, self.root / "pdfs")
        options = RunOptions(report_root=self.root / "reports", download_concurrency=2)
        await run_batch(records, provider, options, recorder)
        for paper_id in ("1", "2"):
            changes = [
                e.result
                for e in recorder.events
                if isinstance(e, PaperChanged) and e.result.record.paper_id == paper_id
            ]
            self.assertEqual(
                [r.status for r in changes],
                [Status.DOWNLOADING, Status.QUEUED, Status.CHECKING, Status.PASSED],
            )
            self.assertIsNone(changes[0].pdf)
            self.assertIsNotNone(changes[1].pdf)  # the waiting row already names its PDF


class StalledDownloadTest(unittest.TestCase):
    def test_cancel_does_not_wait_for_a_stalled_download(self) -> None:
        gate = threading.Event()
        self.addCleanup(gate.set)
        client = FakeClient([note(1), note(2)])
        client.gates["note1"] = gate
        recorder = Recorder()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            async def run() -> tuple[PaperResult, ...]:
                records = await load_openreview(client, VENUE, recorder)
                cancel = asyncio.Event()
                asyncio.get_running_loop().call_later(0.5, cancel.set)
                provider = OpenReviewPdfProvider(client, root / "pdfs")
                options = RunOptions(report_root=root / "reports")
                return await run_batch(records, provider, options, recorder, cancel)

            started = time.monotonic()
            results = asyncio.run(run())
            elapsed = time.monotonic() - started
        # the stalled SDK call runs in a daemon thread, so shutting the loop down skips it
        self.assertLess(elapsed, 5)
        self.assertEqual([r.status for r in results], [Status.CANCELLED, Status.CANCELLED])
        downloading = [
            e.result.record.paper_id
            for e in recorder.events
            if isinstance(e, PaperChanged) and e.result.status is Status.DOWNLOADING
        ]
        self.assertEqual(downloading, ["1"])  # paper 2 never got the only download slot


class LoadTest(OpenReviewTestCase):
    async def test_no_accepted_note_gives_no_records(self) -> None:
        for notes in ([], [note(1, venueid=f"{VENUE}/Rejected_Submission")]):
            client = FakeClient(notes)
            sink = Recorder()
            self.assertEqual(await load_openreview(client, VENUE, sink), [])
            self.assertEqual(client.calls_to("search_profiles"), [])
            self.assertEqual(sink.events, [StageChanged("notes", VENUE)])

    async def test_only_notes_with_the_venue_id_are_accepted(self) -> None:
        notes = [note(3), note(2, venueid=f"{VENUE}/Withdrawn_Submission"), note(1, mdate=42)]
        client = FakeClient(notes)
        sink = Recorder()
        records = await load_openreview(client, VENUE, sink)
        self.assertEqual(client.calls_to("get_all_notes"), [{"content": {"venueid": VENUE}}])
        self.assertEqual([(r.index, r.paper_id) for r in records], [(0, "1"), (1, "3")])
        first = records[0]
        self.assertEqual(
            (first.source, first.source_id, first.file, first.title, first.paper_type),
            (f"openreview:{VENUE}", "note1", "", "Paper 1", "long"),
        )
        self.assertEqual(first.revision, f"/pdf/{1:040x}.pdf@42")
        self.assertEqual(first.problems, ())
        stages = [e.stage for e in sink.events if isinstance(e, StageChanged)]
        self.assertEqual(stages, ["notes", "profiles"])
        last = sink.events[-1]
        assert isinstance(last, StageChanged)
        self.assertIn("ignored 1 notes with another venueid", last.detail)

    async def test_profile_lookups_are_batched_across_papers(self) -> None:
        turing_alias = ("A. Turing", "~Alan_Turing1")
        grace = ("G. Hopper", "Grace@Example.org")
        nobody = ("No Body", "nobody@example.org")
        profiles = (
            FakeProfile(
                "~Ada_Lovelace1",
                {
                    "names": [
                        {"fullname": "Ada Lovelace", "username": "~Ada_Lovelace1"},
                        {"fullname": "Augusta Ada King", "preferred": True},
                    ],
                    "preferredEmail": "ada@example.org",
                    "emails": ["old@example.org", "ada@example.org"],
                },
            ),
            FakeProfile(
                "~Alan_Turing2",
                {
                    "names": [
                        {"first": "Alan", "middle": "Mathison", "last": "Turing"},
                        {"fullname": "A. Turing", "username": "~Alan_Turing1"},
                    ],
                    "emails": ["alan@example.org"],
                },
            ),
            FakeProfile(
                "~Grace_Hopper1",
                {
                    "names": [{"fullname": "Grace Hopper", "username": "~Grace_Hopper1"}],
                    "preferredEmail": "grace@navy.example",
                    "emailsConfirmed": ["grace@example.org"],
                },
            ),
        )
        notes = [
            note(1, authors=[ADA, turing_alias]),
            note(2, authors=[ADA, grace]),
            note(3, authors=[turing_alias, nobody]),
            note(4),
            note(5, authors=[grace, ADA]),
        ]
        client = FakeClient(notes, profiles=profiles)
        records = await load_openreview(client, VENUE, Recorder())
        self.assertEqual(
            client.calls_to("search_profiles"),
            [
                {"ids": ["~Ada_Lovelace1", "~Alan_Turing1"]},
                {"confirmedEmails": ["Grace@Example.org", "nobody@example.org"]},
            ],
        )
        by_id = {r.paper_id: r for r in records}
        self.assertEqual(
            [(a.name, a.email, a.openreview_id) for a in by_id["2"].authors],
            [
                ("Augusta Ada King", "ada@example.org", "~Ada_Lovelace1"),
                ("Grace Hopper", "grace@navy.example", "~Grace_Hopper1"),
            ],
        )
        self.assertEqual(
            [(a.name, a.email, a.openreview_id) for a in by_id["3"].authors],
            [
                ("Alan Mathison Turing", "alan@example.org", "~Alan_Turing2"),
                ("No Body", "nobody@example.org", ""),
            ],
        )
        self.assertEqual(by_id["3"].notes, ("no OpenReview profile found for nobody@example.org",))
        self.assertFalse(any(r.problems for r in records))

    async def test_unified_author_schema_without_emails_makes_one_lookup(self) -> None:
        anonymous = ("Anonymous Coauthor", "")
        client = FakeClient([note(1, authors=[ADA, anonymous], unified=True)])
        (record,) = await load_openreview(client, VENUE, Recorder())
        self.assertEqual(client.calls_to("search_profiles"), [{"ids": ["~Ada_Lovelace1"]}])
        self.assertEqual(
            [(a.name, a.openreview_id) for a in record.authors],
            [("Ada Lovelace", "~Ada_Lovelace1"), ("Anonymous Coauthor", "")],
        )
        self.assertEqual(
            record.notes, ("no OpenReview profile found for ~Ada_Lovelace1", "no author email")
        )

    async def test_unusable_type_or_pdf_field_is_invalid_input(self) -> None:
        notes = [
            note(1, paper_type=None),
            note(2, paper_type="Position Paper"),
            note(3, pdf=""),
            note(4, paper_type="Short Paper (up to 4 pages)"),
        ]
        client = FakeClient(notes)
        results = await self.run_records(client)
        expected = "expected one of long, short, demo, other"
        self.assertEqual(
            [(r.status, r.message) for r in results[:3]],
            [
                (Status.INVALID_INPUT, "missing paper_type"),
                (Status.INVALID_INPUT, f"unknown paper_type 'Position Paper' ({expected})"),
                (Status.INVALID_INPUT, "no PDF in field pdf"),
            ],
        )
        self.assertEqual(results[3].record.paper_type, "short")
        self.assertEqual(results[3].status, Status.PASSED)
        self.assertEqual(results[2].record.revision, "")
        self.assertEqual(client.calls_to("get_pdf"), [{"id": "note4"}])

    async def test_failed_or_missing_profiles_keep_the_paper(self) -> None:
        bob = ("Bob Example", "bob@example.org")
        outage = FakeOpenReviewException({"name": "Error", "message": "Bad gateway", "status": 502})
        client = FakeClient([note(1), note(2, authors=[bob])], profile_errors={"ids": outage})
        results = await self.run_records(client)
        self.assertEqual([r.status for r in results], [Status.PASSED, Status.PASSED])
        failed, missing = (r.record for r in results)
        self.assertEqual([a.name for a in failed.authors], ["Ada Lovelace"])
        self.assertEqual(len(failed.notes), 2)
        self.assertTrue(
            failed.notes[0].startswith(
                "profile lookup failed for ~Ada_Lovelace1: FakeOpenReviewException:"
            ),
            failed.notes,
        )
        self.assertIn("Bad gateway", failed.notes[0])
        self.assertEqual(failed.notes[1], "no author email")
        self.assertEqual(
            [(a.name, a.email) for a in missing.authors], [("Bob Example", "bob@example.org")]
        )
        self.assertEqual(missing.notes, ("no OpenReview profile found for bob@example.org",))


class DownloadTest(OpenReviewTestCase):
    async def test_failed_downloads_never_reach_the_checker(self) -> None:
        stale = self.root / "out" / "pdfs" / "1-0123456789ab.pdf"
        stale.parent.mkdir(parents=True)
        stale.write_bytes(CLEAN)
        rate_limited = FakeOpenReviewException(
            {"name": "RateLimitError", "message": "Too many requests", "status": 429}
        )
        pdfs: dict[str, bytes | Exception] = {
            "note1": ConnectionError("connection reset by peer"),
            "note2": b"<!DOCTYPE html><title>Sign in</title>",
            "note3": rate_limited,
        }
        results = await self.run_records(FakeClient([note(1), note(2), note(3)], pdfs))
        for result, expected in zip(
            results,
            ("ConnectionError: connection reset by peer", "not a PDF: ", "'status': 429"),
        ):
            with self.subTest(result.record.paper_id):
                self.assertEqual(result.status, Status.DOWNLOAD_FAILED)
                self.assertIn(expected, result.message)
                self.assertIsNone(result.pdf)
                self.assertIsNone(result.report_dir)
        self.assertEqual(list(stale.parent.iterdir()), [stale])
        self.assertEqual(stale.read_bytes(), CLEAN)
        self.assertFalse((self.root / "out" / "reports").exists())

    async def test_other_fields_are_attachments_stored_once_per_version(self) -> None:
        client = FakeClient([note(7)])
        client.notes[0].content["camera_ready"] = {"value": "/attachment/abc.pdf"}
        (record,) = await load_openreview(client, VENUE, Recorder(), pdf_field="camera_ready")
        provider = OpenReviewPdfProvider(client, self.root / "pdfs", pdf_field="camera_ready")
        before = datetime.now().astimezone()
        first = await provider.fetch(record)
        second = await provider.fetch(record)
        self.assertEqual(
            client.calls_to("get_attachment"), [{"field_name": "camera_ready", "id": "note7"}] * 2
        )
        self.assertEqual(client.calls_to("get_pdf"), [])
        sha256 = hashlib.sha256(CLEAN).hexdigest()
        self.assertEqual(first.sha256, sha256)
        self.assertEqual(first.path, self.root / "pdfs" / f"7-{sha256[:12]}.pdf")
        self.assertEqual(second.path, first.path)
        self.assertEqual([p.name for p in (self.root / "pdfs").iterdir()], [first.path.name])
        fetched_at = datetime.fromisoformat(first.fetched_at)
        self.assertEqual(fetched_at.utcoffset(), timedelta(0))
        self.assertGreaterEqual(fetched_at, before.replace(microsecond=0))

    async def test_cancel_stops_pending_downloads(self) -> None:
        release = threading.Event()
        self.addCleanup(release.set)  # runs before the directory cleanup, so threads finish
        client = FakeClient([note(1), note(2), note(3)])
        client.gates = {"note2": release, "note3": release}
        records = await load_openreview(client, VENUE, Recorder())
        cancel = asyncio.Event()
        summary = self.root / "summary.csv"
        writer = SummaryWriter(summary)

        def sink(event: Event) -> None:
            writer(event)
            if (
                isinstance(event, PaperChanged)
                and event.result.record.index == 0
                and event.result.status.terminal
            ):
                cancel.set()

        results = await run_batch(
            records,
            OpenReviewPdfProvider(client, self.root / "pdfs"),
            RunOptions(report_root=self.root / "reports"),
            sink,
            cancel,
        )
        self.assertEqual(
            [r.status for r in results], [Status.PASSED, Status.CANCELLED, Status.CANCELLED]
        )
        self.assertEqual(client.calls_to("get_pdf"), [{"id": "note1"}, {"id": "note2"}])
        with summary.open(newline="", encoding="utf8") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(
            [(row["paper_id"], row["status"]) for row in rows],
            [("1", "passed"), ("2", "cancelled"), ("3", "cancelled")],
        )

    async def test_download_concurrency_does_not_change_results(self) -> None:
        notes = [note(n) for n in range(1, 7)]
        pdfs: dict[str, bytes | Exception] = {
            "note1": CLEAN,
            "note2": MARGIN,
            "note3": b"not a pdf",
            "note4": FakeOpenReviewException({"name": "NotFoundError", "status": 404}),
            "note5": CLEAN,
            "note6": MARGIN,
        }
        runs = {}
        for concurrency in (1, 4):
            client = FakeClient(notes, pdfs, delay=0.05)
            results = await self.run_records(
                client, out=f"c{concurrency}", concurrency=concurrency, workers=2
            )
            runs[concurrency] = (semantic(results), client.max_in_flight)
        (serial, serial_peak), (parallel, parallel_peak) = runs[1], runs[4]
        self.assertEqual(serial, parallel)
        self.assertEqual(
            [status for _, status, *_ in serial],
            [
                Status.PASSED,
                Status.VIOLATIONS,
                Status.DOWNLOAD_FAILED,
                Status.DOWNLOAD_FAILED,
                Status.PASSED,
                Status.VIOLATIONS,
            ],
        )
        self.assertEqual(serial_peak, 1)
        self.assertGreater(parallel_peak, 1)


class CliTest(unittest.TestCase):
    """The real aclpubcheck entry point, with only the network client replaced."""

    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.out = Path(directory.name) / "out"

    def run_cli(
        self, client: FakeClient, *args: str
    ) -> tuple[int | None, mock.MagicMock | mock.AsyncMock]:
        argv = ["aclpubcheck", "--openreview-venue", VENUE, "-o", str(self.out), *args]
        with (
            mock.patch.object(sys, "argv", argv),
            mock.patch(
                "aclpubcheck.batch.openreview_source.connect", return_value=client
            ) as connect_mock,
            redirect_stderr(io.StringIO()),
        ):
            code = formatchecker.main()
        return code, connect_mock

    def test_login_failure_is_a_usage_error(self) -> None:
        argv = ["aclpubcheck", "--openreview-venue", VENUE, "-o", str(self.out)]
        stderr = io.StringIO()
        with (
            mock.patch.object(sys, "argv", argv),
            mock.patch(
                "aclpubcheck.batch.openreview_source.connect",
                side_effect=FakeOpenReviewException(
                    {"status": 400, "message": "Invalid credentials"}
                ),
            ),
            redirect_stderr(stderr),
        ):
            code = formatchecker.main()
        self.assertEqual(code, 2)
        self.assertIn("cannot connect to OpenReview", stderr.getvalue())
        self.assertIn("Invalid credentials", stderr.getvalue())
        self.assertFalse(self.out.exists())

    def summary_rows(self) -> list[dict[str, str]]:
        with (self.out / "summary.csv").open(newline="", encoding="utf8") as stream:
            return list(csv.DictReader(stream))

    def test_accepted_papers_are_downloaded_and_checked(self) -> None:
        notes = [note(1), note(2), note(3), note(4, paper_type="Position Paper")]
        pdfs: dict[str, bytes | Exception] = {
            "note1": CLEAN,
            "note2": MARGIN,
            "note3": b"<html>",
            "note4": CLEAN,
        }
        client = FakeClient(notes, pdfs)
        code, connect_mock = self.run_cli(
            client, "--download-concurrency", "2", "--num_workers", "2"
        )
        self.assertEqual(code, 0)
        connect_mock.assert_called_once_with(BASEURL)
        rows = {row["paper_id"]: row for row in self.summary_rows()}
        self.assertEqual(
            {paper_id: row["status"] for paper_id, row in rows.items()},
            {"1": "passed", "2": "violations", "3": "download_failed", "4": "invalid_input"},
        )
        for paper_id, row in rows.items():
            self.assertEqual(
                (row["source"], row["source_id"], row["revision"]),
                (
                    f"openreview:{VENUE}",
                    f"note{paper_id}",
                    f"/pdf/{int(paper_id):040x}.pdf@1700000000000",
                ),
            )
        for paper_id in ("1", "2"):
            row = rows[paper_id]
            served = pdfs[f"note{paper_id}"]
            assert isinstance(served, bytes)
            self.assertEqual(row["pdf_sha256"], hashlib.sha256(served).hexdigest())
            self.assertTrue(row["fetched_at"].endswith("+00:00"), row)
            checked = Path(row["file"])  # the downloaded file, not a name made up from the number
            self.assertTrue(checked.is_file(), row)
            self.assertEqual(checked.name, f"{paper_id}-{row['pdf_sha256'][:12]}.pdf")
            self.assertTrue(Path(row["report_dir"]).is_dir(), row)
        self.assertIn("Margin", rows["2"]["error_categories"])
        self.assertEqual(
            [rows["3"][key] for key in ("pdf_sha256", "fetched_at", "report_dir", "file")],
            ["", "", "", ""],
        )
        self.assertTrue((self.out / "input.json").is_file())

    def test_no_accepted_papers_exit_1_with_an_empty_summary(self) -> None:
        client = FakeClient([note(1, venueid=f"{VENUE}/Rejected_Submission")])
        code, _ = self.run_cli(client)
        self.assertEqual(code, 1)
        lines = (self.out / "summary.csv").read_text(encoding="utf8").splitlines()
        self.assertEqual(lines, [",".join(COLUMNS)])
        self.assertEqual(client.calls_to("get_pdf"), [])

    def test_revised_pdf_keeps_both_versions(self) -> None:
        rows = []
        for pdf, mdate, data in (("/pdf/aaaa.pdf", 1000, CLEAN), ("/pdf/bbbb.pdf", 2000, MARGIN)):
            client = FakeClient([note(1, pdf=pdf, mdate=mdate)], {"note1": data})
            code, _ = self.run_cli(client)
            self.assertEqual(code, 0)
            (row,) = self.summary_rows()
            rows.append(row)
        old, new = rows
        self.assertEqual((old["status"], new["status"]), ("passed", "violations"))
        self.assertEqual(old["source_id"], new["source_id"])
        self.assertEqual(
            (old["revision"], new["revision"]), ("/pdf/aaaa.pdf@1000", "/pdf/bbbb.pdf@2000")
        )
        self.assertNotEqual(old["pdf_sha256"], new["pdf_sha256"])
        self.assertNotEqual(old["report_dir"], new["report_dir"])
        self.assertTrue(Path(old["report_dir"]).is_dir() and Path(new["report_dir"]).is_dir())
        self.assertEqual(
            sorted(p.name for p in (self.out / "pdfs").iterdir()),
            sorted(f"1-{row['pdf_sha256'][:12]}.pdf" for row in rows),
        )


class ConnectTest(unittest.TestCase):
    def fake_sdk(self) -> tuple[dict[str, types.ModuleType], mock.Mock]:
        api = types.ModuleType("openreview.api")
        client_class = mock.Mock(return_value="client")
        setattr(api, "OpenReviewClient", client_class)  # noqa: B010 -- ModuleType has no such attribute to type
        package = types.ModuleType("openreview")
        setattr(package, "api", api)  # noqa: B010 -- as above
        return {"openreview": package, "openreview.api": api}, client_class

    def test_token_comes_from_the_environment(self) -> None:
        modules, client_class = self.fake_sdk()
        with (
            mock.patch.dict(sys.modules, modules),
            mock.patch.dict(os.environ, {"OPENREVIEW_TOKEN": "secret"}),
        ):
            self.assertEqual(connect(BASEURL), "client")
        client_class.assert_called_once_with(baseurl=BASEURL, token="secret")

    def test_without_a_token_the_sdk_reads_username_and_password(self) -> None:
        modules, client_class = self.fake_sdk()
        with mock.patch.dict(sys.modules, modules), mock.patch.dict(os.environ):
            os.environ.pop("OPENREVIEW_TOKEN", None)
            connect(BASEURL)
        client_class.assert_called_once_with(baseurl=BASEURL, token=None)

    def test_missing_sdk_names_the_package(self) -> None:
        with (
            mock.patch.dict(sys.modules, {"openreview": None, "openreview.api": None}),
            self.assertRaisesRegex(MissingDependency, "pip install openreview-py"),
        ):
            connect(BASEURL)

    def test_missing_sdk_is_a_usage_error(self) -> None:
        stderr = io.StringIO()
        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch.dict(sys.modules, {"openreview": None, "openreview.api": None}),
            mock.patch.object(
                sys, "argv", ["aclpubcheck", "--openreview-venue", VENUE, "-o", directory]
            ),
            redirect_stderr(stderr),
        ):
            code = formatchecker.main()
        self.assertEqual(code, 2)
        self.assertIn("pip install openreview-py", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())

    def test_modules_import_without_the_sdk(self) -> None:
        code = (
            "import sys; sys.modules['openreview'] = None; "
            "import aclpubcheck.batch.openreview_source, aclpubcheck.batch.cli"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=60, check=False
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
