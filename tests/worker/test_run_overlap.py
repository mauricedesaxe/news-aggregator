import dagster as dg

from romanian_news.worker.run_overlap import OVERLAP_BLOCKING_RUN_STATUSES


def test_overlap_blocking_run_statuses_are_pinned() -> None:
    assert OVERLAP_BLOCKING_RUN_STATUSES == (
        dg.DagsterRunStatus.QUEUED,
        dg.DagsterRunStatus.NOT_STARTED,
        dg.DagsterRunStatus.MANAGED,
        dg.DagsterRunStatus.STARTING,
        dg.DagsterRunStatus.STARTED,
        dg.DagsterRunStatus.CANCELING,
    )
