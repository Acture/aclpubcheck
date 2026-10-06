"""OpenReview input: one record per accepted paper of a venue, with PDFs downloaded per run."""

from __future__ import annotations

import asyncio
import hashlib
import os
import threading
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from datetime import datetime, timezone
from functools import partial
from itertools import zip_longest
from pathlib import Path
from typing import Protocol, TypeVar, overload

from .model import (
    Author,
    EventSink,
    FetchedPdf,
    FetchError,
    MissingDependency,
    PaperRecord,
    StageChanged,
    Status,
    describe_error,
    metadata_notes,
    normalize_paper_type,
    paper_type_problem,
    text,
    version_name,
    write_atomically,
)

_T = TypeVar("_T")


class Note(Protocol):
    """The fields of an openreview.api.Note this module reads; content values are {"value": ...}."""

    @property
    def id(self) -> str: ...
    @property
    def number(self) -> int: ...
    @property
    def content(self) -> Mapping[str, object]: ...
    @property
    def mdate(self) -> int | None: ...
    @property
    def tmdate(self) -> int | None: ...


class Profile(Protocol):
    @property
    def id(self) -> str: ...
    @property
    def content(self) -> Mapping[str, object]: ...


class OpenReviewClient(Protocol):
    """The calls made on openreview.api.OpenReviewClient; each one is a blocking HTTP request."""

    def get_all_notes(self, *, content: Mapping[str, str]) -> Sequence[Note]: ...

    @overload
    def search_profiles(self, *, ids: list[str]) -> Sequence[Profile]: ...

    @overload
    def search_profiles(self, *, confirmedEmails: list[str]) -> Mapping[str, Profile]: ...

    def get_pdf(self, id: str) -> bytes: ...

    def get_attachment(self, field_name: str, id: str) -> bytes: ...


async def call_sdk(call: Callable[[], _T]) -> _T:
    """Run one blocking SDK call in a daemon thread.

    asyncio.to_thread's threads are joined when the loop and the interpreter shut down, and
    the SDK sets no request timeout and retries 429s for minutes, so a cancelled run would
    wait for a stalled request. A daemon thread is abandoned instead."""
    loop = asyncio.get_running_loop()
    future: asyncio.Future[_T] = loop.create_future()

    def settle(outcome: Callable[[], object]) -> None:
        if not future.done():  # the awaiting task may have been cancelled
            outcome()

    def work() -> None:
        try:
            result = call()
        except BaseException as error:  # noqa: BLE001 -- re-raised in the awaiting task
            deliver = partial(settle, partial(future.set_exception, error))
        else:
            deliver = partial(settle, partial(future.set_result, result))
        with suppress(RuntimeError):  # the loop closed after the run was cancelled
            loop.call_soon_threadsafe(deliver)

    threading.Thread(target=work, name="openreview-sdk", daemon=True).start()
    return await future


def connect(baseurl: str) -> OpenReviewClient:
    """An API v2 client; credentials come from the environment only, never the command line.

    OPENREVIEW_TOKEN is used when set; otherwise the SDK itself logs in with
    OPENREVIEW_USERNAME/OPENREVIEW_PASSWORD, and with neither it stays anonymous and sees
    only public notes.
    """
    try:
        import openreview.api
    except ImportError as error:
        raise MissingDependency(
            "--openreview-venue needs openreview-py: pip install openreview-py"
        ) from error
    return openreview.api.OpenReviewClient(
        baseurl=baseurl, token=os.environ.get("OPENREVIEW_TOKEN") or None
    )


def _value(content: Mapping[str, object], key: str) -> object:
    field = content.get(key)
    return field.get("value") if isinstance(field, Mapping) else None


def _items(value: object) -> list[object]:
    return list(value) if isinstance(value, list) else []


def is_accepted(note: Note, venue_id: str) -> bool:
    """A note is an accepted paper of the venue when its venueid is exactly the venue id.

    Publication chairs cannot read decision replies, so the venueid set on acceptance is the
    only selection they can rely on (as in aclpub2's openreview/or2papers.py).
    """
    return _value(note.content, "venueid") == venue_id


def _note_authors(content: Mapping[str, object]) -> list[tuple[str, str]]:
    """(name shown on the note, author id) per author, for both OpenReview author schemas."""
    authors = _items(_value(content, "authors"))
    if authors and isinstance(authors[0], Mapping):  # unified schema: [{fullname, username}]
        return [
            (text(a.get("fullname")), text(a.get("username")))
            for a in authors
            if isinstance(a, Mapping)
        ]
    ids = _items(_value(content, "authorids"))
    return [(text(name), text(author_id)) for name, author_id in zip_longest(authors, ids)]


def _usernames(profile: Profile) -> list[str]:
    """Every id the profile answers to: a merged or renamed profile keeps its old usernames."""
    names = (n for n in _items(profile.content.get("names")) if isinstance(n, Mapping))
    return [profile.id, *(text(n.get("username")) for n in names if n.get("username"))]


def _preferred_name(profile: Profile) -> str:
    """aclpub2's get_user: the last name entry marked preferred, else the first entry."""
    names = [n for n in _items(profile.content.get("names")) if isinstance(n, Mapping)]
    if not names:
        return ""
    chosen = next((n for n in reversed(names) if n.get("preferred")), names[0])
    parts = (text(chosen.get(key)) for key in ("first", "middle", "last"))
    return text(chosen.get("fullname")) or " ".join(part for part in parts if part)


def _preferred_email(profile: Profile) -> str:
    """aclpub2's get_user: the preferred email, else the first listed one."""
    emails = _items(profile.content.get("emails"))
    return text(profile.content.get("preferredEmail")) or (text(emails[0]) if emails else "")


async def _lookup(
    keys: list[str], search: Callable[[], _T], index: Callable[[_T], Mapping[str, Profile]]
) -> tuple[Mapping[str, Profile], str]:
    """Profiles by author id from one batched search, or why the search failed."""
    if not keys:  # the SDK answers an empty search with a list, whatever was asked
        return {}, ""
    try:
        found = await call_sdk(search)
    except Exception as error:  # noqa: BLE001 -- these authors fall back to the note's names
        return {}, describe_error(error)
    return index(found), ""


def _author(
    position: int,
    shown_name: str,
    author_id: str,
    profiles: Mapping[str, Profile],
    failures: Mapping[str, str],
    notes: list[str],
) -> Author:
    is_email = "@" in author_id
    profile = profiles.get(author_id.lower() if is_email else author_id)
    if author_id in failures:
        notes.append(f"profile lookup failed for {author_id}: {failures[author_id]}")
    elif profile is None and author_id:
        notes.append(f"no OpenReview profile found for {author_id}")
    fallback_email = author_id if is_email else ""
    if profile is None:
        author = Author(
            name=shown_name,
            email=fallback_email,
            openreview_id=author_id if author_id.startswith("~") else "",
        )
    else:
        author = Author(
            name=_preferred_name(profile) or shown_name,
            email=_preferred_email(profile) or fallback_email,
            openreview_id=profile.id,
        )
    if not author.name:
        notes.append(f"author {position} has no name")
    return author


def _record(
    index: int,
    note: Note,
    pdf_field: str,
    profiles: Mapping[str, Profile],
    failures: Mapping[str, str],
    source: str,
) -> PaperRecord:
    content = note.content
    problems: list[str] = []
    notes: list[str] = []
    raw_type = _value(content, "paper_type")
    if problem := paper_type_problem(raw_type, "paper_type"):
        problems.append(problem)
    pdf = text(_value(content, pdf_field))
    if not pdf:
        problems.append(f"no PDF in field {pdf_field}")
    title = text(_value(content, "title"))
    authors = tuple(
        _author(position, name, author_id, profiles, failures, notes)
        for position, (name, author_id) in enumerate(_note_authors(content), 1)
    )
    notes += metadata_notes(title, authors)
    stamp = note.mdate or note.tmdate
    return PaperRecord(
        index=index,
        paper_id=str(note.number),
        title=title,
        paper_type=normalize_paper_type(raw_type),
        authors=authors,
        source=source,
        source_id=note.id,
        revision=f"{pdf}@{stamp}" if pdf and stamp is not None else pdf,
        problems=tuple(problems),
        notes=tuple(notes),
    )


async def load_openreview(
    client: OpenReviewClient, venue_id: str, sink: EventSink, *, pdf_field: str = "pdf"
) -> list[PaperRecord]:
    """Records for the accepted papers of `venue_id`, by note number; bad notes carry problems.

    - Selection: get_all_notes(content={"venueid": venue_id}), then only notes passing
      is_accepted. No accepted note gives [].
    - Record: paper_id is the note number, source "openreview:<venue_id>", source_id the note
      id, no declared file (the downloaded PDF is named by the provider), and revision
      "<pdf_field value>@<mdate, else tmdate>": the value names the uploaded file, so it
      changes with every upload and is known before downloading.
    - Problems (never checked): a missing or unrecognised paper_type, which is never
      defaulted, and an empty pdf_field.
    - Authors: every author id of every note is looked up once, tilde ids in one
      search_profiles(ids=...) call and emails in one search_profiles(confirmedEmails=...)
      call. Name and email follow aclpub2's get_user. A failed or empty lookup keeps the
      paper, uses the name shown on the note, and adds a note to the record.
    - Every SDK call runs in a daemon thread (call_sdk), so the event loop never blocks and a
      cancelled run does not wait for a stalled request.
    """
    sink(StageChanged("notes", venue_id))
    notes = await call_sdk(lambda: client.get_all_notes(content={"venueid": venue_id}))
    accepted = sorted((n for n in notes if is_accepted(n, venue_id)), key=lambda n: n.number)
    if not accepted:
        return []
    author_ids = list(
        dict.fromkeys(
            author_id
            for note in accepted
            for _, author_id in _note_authors(note.content)
            if author_id
        )
    )
    tilde_ids = [i for i in author_ids if i.startswith("~")]
    emails = [i for i in author_ids if "@" in i]
    skipped = len(notes) - len(accepted)
    sink(
        StageChanged(
            "profiles",
            f"{len(author_ids)} authors of {len(accepted)} accepted papers"
            + (f"; ignored {skipped} notes with another venueid" if skipped else ""),
        )
    )
    by_id, id_failure = await _lookup(
        tilde_ids,
        lambda: client.search_profiles(ids=tilde_ids),
        lambda found: {name: p for p in found for name in _usernames(p)},
    )
    by_email, email_failure = await _lookup(
        emails,
        lambda: client.search_profiles(confirmedEmails=emails),
        lambda found: {email.lower(): p for email, p in found.items()},
    )
    profiles = {**by_email, **by_id}
    failures = {i: id_failure for i in tilde_ids if id_failure} | {
        e: email_failure for e in emails if email_failure
    }
    return [
        _record(index, note, pdf_field, profiles, failures, f"openreview:{venue_id}")
        for index, note in enumerate(accepted)
    ]


def _store(directory: Path, paper_id: str, data: bytes) -> tuple[Path, str]:
    """Write `data` atomically under a name unique to the paper and its bytes.

    An existing file of that name is replaced rather than trusted, so the checked bytes are
    always the ones just downloaded."""
    sha256 = hashlib.sha256(data).hexdigest()
    path = directory / f"{version_name(paper_id, sha256)}.pdf"
    write_atomically(path, data)
    return path, sha256


class OpenReviewPdfProvider:
    """Downloads each record's PDF from its note into `download_dir`.

    - Field "pdf" is fetched with get_pdf(note id), any other field with
      get_attachment(field_name=field, id=note id).
    - The runner decides how many downloads overlap (RunOptions.download_concurrency, 1 by
      default, as OpenReview documents no rate limit); concurrent downloads share the
      client's one requests.Session across threads.
    - The SDK already retries 429 and 5xx responses honouring Retry-After, so no retry is added
      here: once it gives up, or the bytes are not a PDF, the paper ends as download_failed.
    - The bytes are written atomically to "<paper id>-<sha256[:12]>.pdf", so every version is
      kept. A failed download never falls back to a PDF left by an earlier run.
    """

    remote = True

    def __init__(
        self,
        client: OpenReviewClient,
        download_dir: Path,
        *,
        pdf_field: str = "pdf",
    ) -> None:
        self.client = client
        self.download_dir = download_dir
        self.pdf_field = pdf_field

    def _download(self, note_id: str) -> bytes:
        if self.pdf_field == "pdf":
            return self.client.get_pdf(note_id)
        return self.client.get_attachment(field_name=self.pdf_field, id=note_id)

    async def fetch(self, record: PaperRecord) -> FetchedPdf:
        try:
            data = await call_sdk(lambda: self._download(record.source_id))
        except Exception as error:
            raise FetchError(Status.DOWNLOAD_FAILED, describe_error(error)) from error
        fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        if not data.startswith(b"%PDF-"):
            raise FetchError(
                Status.DOWNLOAD_FAILED, f"not a PDF: {len(data)} bytes starting {data[:16]!r}"
            )
        path, sha256 = await asyncio.to_thread(_store, self.download_dir, record.paper_id, data)
        return FetchedPdf(path=path, sha256=sha256, fetched_at=fetched_at)
