from datetime import datetime

from romanian_news.articles import recovery
from romanian_news.articles.models import ArticleAcquisitionFailure, ArticleFailureKind
from romanian_news.catalog.articles import ArticleFailureAttempt

EVENT_ID = "a" * 64
FIRST = "b" * 64
SECOND = "c" * 64
FIRST_GENERATION = "d" * 64
SECOND_GENERATION = "e" * 64


def test_unchanged_deterministic_failures_quarantine_after_three_attempts(monkeypatch) -> None:
    monkeypatch.setattr(
        recovery.article_catalog,
        "read_article_failure_attempts",
        lambda *_args: (
            _row("deterministic", FIRST, "2026-09-09T10:00:00+00:00"),
            _row("infrastructure", SECOND, "2026-09-09T10:05:00+00:00"),
            _row("deterministic", FIRST, "2026-09-09T10:10:00+00:00"),
            _row("deterministic", FIRST, "2026-09-09T10:20:00+00:00"),
        ),
    )

    state = recovery.read_article_attempt_states({EVENT_ID: FIRST_GENERATION})[EVENT_ID]

    assert state.quarantined
    assert state.unchanged_deterministic_attempts == 3


def test_changed_deterministic_failure_resets_poison_count(monkeypatch) -> None:
    monkeypatch.setattr(
        recovery.article_catalog,
        "read_article_failure_attempts",
        lambda *_args: (
            _row("deterministic", FIRST, "2026-09-09T10:00:00+00:00"),
            _row("deterministic", FIRST, "2026-09-09T10:10:00+00:00"),
            _row("deterministic", SECOND, "2026-09-09T10:20:00+00:00"),
        ),
    )

    state = recovery.read_article_attempt_states({EVENT_ID: FIRST_GENERATION})[EVENT_ID]

    assert not state.quarantined
    assert state.deterministic_fingerprint == SECOND
    assert state.unchanged_deterministic_attempts == 1
    assert state.deferred(datetime.fromisoformat("2026-09-09T10:15:00+00:00"))


def test_fail_fail_success_fail_counts_the_last_failure_as_one(monkeypatch) -> None:
    success_at = datetime.fromisoformat("2026-09-09T11:00:00+00:00")
    before_success = recovery.article_work_generation(EVENT_ID, None)
    after_success = recovery.article_work_generation(EVENT_ID, success_at)
    monkeypatch.setattr(
        recovery.article_catalog,
        "read_article_failure_attempts",
        lambda *_args: (
            _row(
                "deterministic",
                FIRST,
                "2026-09-09T10:00:00+00:00",
                work_generation=before_success,
            ),
            _row(
                "deterministic",
                FIRST,
                "2026-09-09T10:10:00+00:00",
                work_generation=before_success,
            ),
            _row(
                "deterministic",
                FIRST,
                "2026-09-09T11:15:00+00:00",
                work_generation=after_success,
            ),
        ),
    )

    state = recovery.read_article_attempt_states({EVENT_ID: after_success})[EVENT_ID]

    assert not state.quarantined
    assert state.work_generation == after_success
    assert state.unchanged_deterministic_attempts == 1


def test_quarantine_survives_a_deploy_for_the_same_failure(monkeypatch) -> None:
    rows = [
        _row(
            "deterministic",
            FIRST,
            f"2026-09-09T10:{minute:02d}:00+00:00",
            implementation_ref="git:old",
        )
        for minute in (0, 10)
    ] + [
        _row(
            "deterministic",
            FIRST,
            "2026-09-09T11:30:00+00:00",
            implementation_ref="git:new",
        )
    ]

    def read_attempts(event_ids):
        assert event_ids == (EVENT_ID,)
        return tuple(rows)

    monkeypatch.setattr(
        recovery.article_catalog,
        "read_article_failure_attempts",
        read_attempts,
    )

    state = recovery.read_article_attempt_states({EVENT_ID: FIRST_GENERATION})[EVENT_ID]

    assert state.quarantined
    assert state.unchanged_deterministic_attempts == 3
    assert state.implementation_ref == "git:new"


def test_new_implementation_failure_resets_old_quarantine(monkeypatch) -> None:
    rows = [
        _row(
            "deterministic",
            FIRST,
            f"2026-09-09T10:{minute:02d}:00+00:00",
            implementation_ref="git:old",
        )
        for minute in (0, 10, 20)
    ] + [
        _row(
            "deterministic",
            SECOND,
            "2026-09-09T11:30:00+00:00",
            implementation_ref="git:new",
        )
    ]

    def read_attempts(event_ids):
        assert event_ids == (EVENT_ID,)
        return tuple(rows)

    monkeypatch.setattr(
        recovery.article_catalog,
        "read_article_failure_attempts",
        read_attempts,
    )

    state = recovery.read_article_attempt_states({EVENT_ID: FIRST_GENERATION})[EVENT_ID]

    assert not state.quarantined
    assert state.deterministic_fingerprint == SECOND
    assert state.unchanged_deterministic_attempts == 1


def test_failure_attempt_insert_records_unambiguous_identity(monkeypatch) -> None:
    writes = []
    monkeypatch.setattr(recovery, "read_article_attempt_states", lambda *_args: {})
    monkeypatch.setattr(
        recovery.article_catalog,
        "write_article_failure_attempts",
        writes.append,
    )
    failure = ArticleAcquisitionFailure(
        event_id=EVENT_ID,
        kind=ArticleFailureKind.DETERMINISTIC,
        fingerprint=FIRST,
        message="invalid article",
    )

    recovery.record_article_failure_attempts(
        (failure,),
        work_generations={EVENT_ID: FIRST_GENERATION},
        run_id="run-1",
        retry_number=0,
        implementation_ref="git:current",
        attempted_at=datetime.fromisoformat("2026-09-09T09:00:00+00:00"),
    )

    attempt = writes[0][0]
    assert attempt.event_id == EVENT_ID
    assert attempt.implementation_ref == "git:current"
    assert attempt.work_generation == FIRST_GENERATION
    assert attempt.dagster_run_id == "run-1"
    assert attempt.retry_number == 0


def _row(
    kind: str,
    fingerprint: str,
    retry_at: str,
    *,
    implementation_ref: str = "git:current",
    work_generation: str = FIRST_GENERATION,
) -> ArticleFailureAttempt:
    return ArticleFailureAttempt(
        event_id=EVENT_ID,
        implementation_ref=implementation_ref,
        work_generation=work_generation,
        failure_kind=ArticleFailureKind(kind),
        failure_fingerprint=fingerprint,
        retry_at=datetime.fromisoformat(retry_at),
    )
