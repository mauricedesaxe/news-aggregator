from __future__ import annotations

from typing import final

from romanian_news.catalog.report_inputs import read_exact_daily_report_file
from romanian_news.catalog.video_digest import read_edition_identity
from romanian_news.reports import DailyReport
from romanian_news.storage import read_verified_r2_object
from romanian_news.video_digest.models import EditionId, SlotId, SlotSkipReason
from romanian_news.video_digest.orchestration import (
    ActionOutcome,
    AlertDisposition,
    CatalogPort,
    DomainPorts,
    GenerationPort,
    PlanAction,
    ScheduledResume,
    SourceUnavailable,
    VideoDigestRunOutcome,
    VideoDigestRunRequest,
    run_video_digest,
)
from romanian_news.video_digest.preflight import PlanningReport
from romanian_news.video_digest.publication import R2PublicationPort
from romanian_news.video_digest.subtitle_port import ResumableSubtitlePort
from romanian_news.worker.video_digest_assembly import ProductionAssemblyPort
from romanian_news.worker.video_digest_catalog import PostgresVideoDigestCatalog
from romanian_news.worker.video_digest_planning import ProductionPlanningPort
from romanian_news.worker.video_digest_source import resolve_video_digest_run_request


@final
class _ExactPlanningPort:
    def execute(self, action: PlanAction) -> ActionOutcome:
        edition_id: EditionId = action.lease.edition_id
        identity = read_edition_identity(edition_id)
        file = read_exact_daily_report_file(identity.daily_report_version_id)
        content = read_verified_r2_object(file.r2_key, file.content_digest)
        report = DailyReport.model_validate_json(content, strict=True)
        return ProductionPlanningPort(
            PlanningReport(version_id=identity.daily_report_version_id, report=report)
        ).execute(action)


@final
class ProductionVideoDigestRuntime:
    def __init__(
        self,
        generation: GenerationPort,
        *,
        catalog: CatalogPort | None = None,
        ports: DomainPorts | None = None,
    ) -> None:
        self._catalog: CatalogPort = catalog or PostgresVideoDigestCatalog()
        self._ports: DomainPorts = ports or DomainPorts(
            planning=_ExactPlanningPort(),
            generation=generation,
            assembly=ProductionAssemblyPort(),
            subtitles=ResumableSubtitlePort(),
            publication=R2PublicationPort.from_environment(),
        )

    def run(
        self, slot_id: SlotId, *, owner_token: str
    ) -> tuple[VideoDigestRunOutcome, AlertDisposition]:
        state = self._catalog.read(slot_id)
        if isinstance(state, ScheduledResume):
            request = resolve_video_digest_run_request(state.slot, owner_token=owner_token)
        else:
            request = VideoDigestRunRequest(
                slot=state.slot,
                owner_token=owner_token,
                source=SourceUnavailable(reason=SlotSkipReason.SOURCE_MISSING),
            )
        return run_video_digest(request, self._catalog, self._ports)
