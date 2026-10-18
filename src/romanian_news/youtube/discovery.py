from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import ClassVar, Literal

import requests
from pydantic import ConfigDict, HttpUrl, ValidationError

from romanian_news import NewsModel, Sha256
from romanian_news.config import YOUTUBE_API_KEY
from romanian_news.youtube.errors import (
    YouTubeConfigurationError,
    YouTubeInfrastructureError,
    YouTubeProviderPayloadError,
)
from romanian_news.youtube.models import (
    VideoId,
    YouTubeFeed,
    YouTubeFeedEntry,
    YouTubePollCapture,
    YouTubePollDecision,
    YouTubePollStatus,
    YouTubeSource,
    YouTubeVideoId,
    YouTubeVideoMetadata,
)

_YOUTUBE_VIDEOS_API = "https://www.googleapis.com/youtube/v3/videos"
_YOUTUBE_PLAYLIST_ITEMS_API = "https://www.googleapis.com/youtube/v3/playlistItems"


class _YouTubeProviderModel(NewsModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore")


class _YouTubeContentDetails(_YouTubeProviderModel):
    duration: str


class _YouTubeSnippet(_YouTubeProviderModel):
    title: str
    publishedAt: datetime
    channelId: str


class _YouTubeVideo(_YouTubeProviderModel):
    id: str
    contentDetails: _YouTubeContentDetails
    snippet: _YouTubeSnippet


class _YouTubeResponse(_YouTubeProviderModel):
    kind: Literal["youtube#videoListResponse"]
    items: tuple[_YouTubeVideo, ...]


class _YouTubePlaylistResource(_YouTubeProviderModel):
    videoId: str


class _YouTubePlaylistSnippet(_YouTubeProviderModel):
    channelId: str
    title: str
    resourceId: _YouTubePlaylistResource


class _YouTubePlaylistContent(_YouTubeProviderModel):
    videoPublishedAt: datetime


class _YouTubePlaylistItem(_YouTubeProviderModel):
    snippet: _YouTubePlaylistSnippet
    contentDetails: _YouTubePlaylistContent


class _YouTubePlaylistResponse(_YouTubeProviderModel):
    kind: Literal["youtube#playlistItemListResponse"]
    items: tuple[_YouTubePlaylistItem, ...]


def parse_youtube_feed(source: YouTubeSource, content: bytes) -> YouTubeFeed:
    try:
        payload = _YouTubePlaylistResponse.model_validate_json(content, strict=True)
    except (ValidationError, ValueError) as error:
        raise YouTubeProviderPayloadError("YouTube feed response is invalid") from error
    if any(item.snippet.channelId != source.channel_id for item in payload.items):
        raise YouTubeProviderPayloadError("YouTube feed belongs to another channel")
    try:
        return YouTubeFeed(
            source=source,
            entries=tuple(
                YouTubeFeedEntry(
                    source=source,
                    video_id=YouTubeVideoId(item.snippet.resourceId.videoId),
                    title=item.snippet.title,
                    published_at=item.contentDetails.videoPublishedAt,
                )
                for item in payload.items
            ),
        )
    except ValidationError as error:
        raise YouTubeProviderPayloadError("YouTube feed response is invalid") from error


def decide_youtube_poll(
    feed: YouTubeFeed, known_video_ids: frozenset[str], *, baseline_exists: bool
) -> YouTubePollDecision:
    current = tuple(entry.video_id for entry in feed.entries)
    if not baseline_exists:
        return YouTubePollDecision(
            source=feed.source, status=YouTubePollStatus.BASELINE, pending_video_ids=()
        )
    if known_video_ids and not known_video_ids.intersection(current):
        return YouTubePollDecision(
            source=feed.source, status=YouTubePollStatus.OVERLAP_LOST, pending_video_ids=()
        )
    return YouTubePollDecision(
        source=feed.source,
        status=YouTubePollStatus.CURRENT,
        pending_video_ids=tuple(value for value in current if value not in known_video_ids),
    )


def acquire_youtube_poll(source: YouTubeSource, scheduled_at: datetime) -> YouTubePollCapture:
    if scheduled_at.tzinfo is None:
        raise ValueError("YouTube poll time must include a timezone")
    content = _request_youtube(
        _YOUTUBE_PLAYLIST_ITEMS_API,
        {
            "part": "snippet,contentDetails",
            "playlistId": source.uploads_playlist_id,
            "maxResults": 50,
        },
        failure_message="YouTube feed request failed",
    )
    return YouTubePollCapture(
        source=source,
        scheduled_at=scheduled_at,
        observed_at=datetime.now(UTC),
        content=content,
        feed=parse_youtube_feed(source, content),
    )


def fetch_video_metadata(
    source: YouTubeSource, video_id: VideoId, feed_snapshot_version_id: Sha256
) -> YouTubeVideoMetadata:
    raw_content = _request_youtube(
        _YOUTUBE_VIDEOS_API,
        {"part": "snippet,contentDetails", "id": video_id},
        failure_message="YouTube metadata request failed",
    )
    try:
        payload = _YouTubeResponse.model_validate_json(raw_content, strict=True)
    except (ValidationError, ValueError) as error:
        raise YouTubeProviderPayloadError("YouTube metadata response is invalid") from error
    if len(payload.items) != 1 or payload.items[0].id != video_id:
        raise YouTubeProviderPayloadError("YouTube metadata did not return the requested video")
    video = payload.items[0]
    if video.snippet.channelId != source.channel_id:
        raise YouTubeProviderPayloadError("YouTube metadata belongs to another channel")
    return YouTubeVideoMetadata(
        source=source,
        video_id=YouTubeVideoId(video.id),
        url=HttpUrl(f"https://www.youtube.com/watch?v={video.id}"),
        title=video.snippet.title,
        published_at=video.snippet.publishedAt,
        duration_seconds=_parse_iso_duration(video.contentDetails.duration),
        feed_snapshot_version_id=feed_snapshot_version_id,
        retrieved_at=datetime.now(UTC),
        provider_response=payload.model_dump(mode="json"),
        raw_provider_response=raw_content,
    )


def _request_youtube(
    endpoint: str,
    params: dict[str, str | int],
    *,
    failure_message: str,
) -> bytes:
    if not YOUTUBE_API_KEY:
        raise YouTubeConfigurationError("YOUTUBE_API_KEY is required")
    try:
        response = requests.get(
            endpoint,
            headers={"X-Goog-Api-Key": YOUTUBE_API_KEY},
            params=params,
            timeout=30,
        )
        response.raise_for_status()
    except requests.RequestException as error:
        raise YouTubeInfrastructureError(failure_message) from error
    return response.content


def _parse_iso_duration(value: str) -> int:
    match = re.fullmatch(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", value)
    if match is None:
        raise YouTubeProviderPayloadError(f"YouTube duration is invalid: {value}")
    hours, minutes, seconds = (int(part or 0) for part in match.groups())
    duration = hours * 3600 + minutes * 60 + seconds
    if duration <= 0:
        raise YouTubeProviderPayloadError("YouTube duration must be positive")
    return duration
