#!/usr/bin/env python3
"""Build a consumer-focused AI model timeline from public Polymarket data.

The script uses only Python's standard library. It discovers current markets via
Polymarket search, tag, and event-list APIs, fetches canonical records, and
handles these market shapes:

* cumulative deadlines ("released by ...") -> median release estimate;
* mutually exclusive date buckets ("released on ...") -> expected/median date;
* binary consumer changes (access, pricing, retirement) -> probability/deadline;
* multi-deadline consumer changes -> cumulative forecast or resolved outcome.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import datetime as dt
import hashlib
import html
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"
POLYMARKET_EVENT_URL = "https://polymarket.com/event/{}"
USER_AGENT = "ai-model-release-timeline/1.0"
SITE_TITLE = "AI Model Release Timeline — Prediction Market Forecasts"
SITE_DESCRIPTION = (
    "A market-implied timeline of upcoming OpenAI, Anthropic, and Google Gemini model releases "
    "and consumer-facing changes."
)

# These are discovery terms, not a fixed market list. Add a term when a provider
# starts using a new product family name. --query and --slug are also available.
DEFAULT_SEARCH_QUERIES = (
    "OpenAI",
    "ChatGPT",
    "GPT",
    "Anthropic",
    "Claude",
    "Gemini",
    "GPT-6",
    "OpenAI model release",
    "OpenAI access pricing",
    "ChatGPT access",
    "Anthropic Claude",
    "Claude model",
    "Claude access",
    "Claude pricing",
    "Google Gemini",
    "Gemini model",
    "Gemini access pricing",
)

DEFAULT_TAG_SLUGS = ("openai", "claude", "gemini-ultra")
CLOSED_SEARCH_QUERIES = ("OpenAI", "ChatGPT", "Anthropic", "Claude", "Gemini")
DISCOVERY_STATE_VERSION = 1
INCREMENTAL_SCAN_PAGE_LIMIT = 50

PROVIDER_TERMS = {
    "OpenAI": ("openai", "chatgpt", "gpt-"),
    "Anthropic": ("anthropic", "claude", "opus", "sonnet", "haiku", "mythos", "fable"),
    "Google": ("google", "gemini"),
}

CONSUMER_TERMS = (
    "release",
    "launch",
    "available",
    "availability",
    "access",
    "paid plan",
    "usage credit",
    "pricing",
    "price",
    "retire",
    "retirement",
    "deprecat",
    "extend",
    "context window",
)

EXCLUDED_TITLE_TERMS = (
    "best ai model",
    "which company has the best",
    "#1 ai model",
    "#2 ai model",
    "#3 ai model",
    "arena score",
    "arena debut",
    "valuation",
    "market cap",
    " ipo",
    "acquired",
    "app store",
    "consumer hardware",
    "social network",
    "launch a token",
    "go down on",
    "before gta",
    "trump orders",
    "government removes",
)

MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}
MONTH_PATTERN = "|".join(MONTHS)
DATE_RE = re.compile(
    rf"\b({MONTH_PATTERN})\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:,?\s+(20\d{{2}}))?\b",
    re.IGNORECASE,
)


class ApiError(RuntimeError):
    pass


@dataclasses.dataclass(frozen=True)
class ProbabilityPoint:
    date: dt.date
    probability: float
    label: str
    volume: float


@dataclasses.dataclass(frozen=True)
class DistributionPoint:
    date: dt.date
    cdf: float
    pdf: float


@dataclasses.dataclass(frozen=True)
class MarketSource:
    label: str
    title: str
    url: str
    event_id: str


@dataclasses.dataclass(frozen=True)
class TimelineItem:
    provider: str
    title: str
    kind: str
    sort_date: dt.date
    when: str
    estimate: str
    detail: str
    volume: float
    url: str
    event_id: str
    distribution: tuple[DistributionPoint, ...] = ()
    sources: tuple[MarketSource, ...] = ()
    median_date: dt.date | None = None

    def as_dict(self) -> dict[str, Any]:
        result = dataclasses.asdict(self)
        result["sort_date"] = self.sort_date.isoformat()
        result["median_date"] = self.median_date.isoformat() if self.median_date else None
        result["distribution"] = [
            {"date": point.date.isoformat(), "cdf": point.cdf, "pdf": point.pdf} for point in self.distribution
        ]
        return result


@dataclasses.dataclass(frozen=True)
class HistorySnapshot:
    date: dt.date
    items: tuple[TimelineItem, ...]


@dataclasses.dataclass(frozen=True)
class DiscoveryDecision:
    event_id: str
    slug: str
    title: str
    provider: str | None
    shape: str
    classification: str
    change: str
    reason: str
    sources: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class DiscoveryResult:
    events: tuple[dict[str, Any], ...]
    decisions: tuple[DiscoveryDecision, ...]
    state: dict[str, Any]
    warnings: tuple[str, ...] = ()
    history_events: tuple[dict[str, Any], ...] = ()


class StateError(RuntimeError):
    pass


def get_json(path: str, params: dict[str, Any] | None = None, retries: int = 3) -> Any:
    query = urllib.parse.urlencode(params or {}, doseq=True)
    url = f"{GAMMA_API}{path}" + (f"?{query}" if query else "")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                return json.load(response)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            if attempt + 1 == retries:
                raise ApiError(f"GET {url} failed: {exc}") from exc
            time.sleep(0.5 * (2**attempt))
    raise AssertionError("unreachable")


def post_clob_json(path: str, body: dict[str, Any], retries: int = 3) -> Any:
    url = f"{CLOB_API}{path}"
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"User-Agent": USER_AGENT, "Accept": "application/json", "Content-Type": "application/json"},
        method="POST",
    )
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.load(response)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            if attempt + 1 == retries:
                raise ApiError(f"POST {url} failed: {exc}") from exc
            time.sleep(0.5 * (2**attempt))
    raise AssertionError("unreachable")


def default_discovery_state_path() -> Path:
    return Path.home() / ".polymarket-model-timeline" / "discovery-state.json"


def empty_discovery_state() -> dict[str, Any]:
    return {"version": DISCOVERY_STATE_VERSION, "max_event_id": 0, "events": {}}


def load_discovery_state(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return empty_discovery_state()
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StateError(f"Cannot read discovery state {path}: {exc}") from exc
    if not isinstance(state, dict) or state.get("version") != DISCOVERY_STATE_VERSION:
        raise StateError(f"Unsupported discovery state format in {path}")
    if not isinstance(state.get("events"), dict):
        raise StateError(f"Invalid discovery event records in {path}")
    return state


def save_discovery_state(path: Path | None, state: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def numeric_event_id(event: dict[str, Any]) -> int:
    try:
        return int(event.get("id") or 0)
    except (TypeError, ValueError):
        return 0


def candidate_mentions_provider(event: dict[str, Any]) -> bool:
    market_questions = " ".join(str(market.get("question") or "") for market in event.get("markets") or [])
    text = f"{event.get('title', '')} {market_questions} {event.get('description', '')}".lower()
    return any(term in text for terms in PROVIDER_TERMS.values() for term in terms)


def discovery_fingerprint(event: dict[str, Any]) -> str:
    relevant = {
        "active": event.get("active"),
        "closed": event.get("closed"),
        "title": event.get("title"),
        "description": event.get("description"),
        "markets": [
            {
                "id": market.get("id"),
                "question": market.get("question"),
                "active": market.get("active"),
                "closed": market.get("closed"),
                "acceptingOrders": market.get("acceptingOrders"),
                "endDate": market.get("endDate"),
            }
            for market in event.get("markets") or []
        ],
    }
    encoded = json.dumps(relevant, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def discover_event_candidates(
    queries: Sequence[str],
    extra_slugs: Sequence[str],
    state: dict[str, Any] | None = None,
    tag_slugs: Sequence[str] = DEFAULT_TAG_SLUGS,
) -> DiscoveryResult:
    """Collect, classify, and audit canonical events from several discovery feeds."""
    previous_state = state or empty_discovery_state()
    summaries: dict[str, dict[str, Any]] = {}
    source_names: dict[str, set[str]] = {}
    warnings: list[str] = []

    def add(event: dict[str, Any], source: str) -> None:
        slug = str(event.get("slug") or "")
        if not slug:
            return
        summaries[slug] = event
        source_names.setdefault(slug, set()).add(source)

    for query in queries:
        for page in range(1, 4):
            data = get_json(
                "/public-search",
                {
                    "q": query,
                    "events_status": "active",
                    "limit_per_type": 50,
                    "page": page,
                    "keep_closed_markets": 0,
                    "search_profiles": "false",
                },
            )
            for event in data.get("events") or []:
                add(event, f"search:{query}")
            if not (data.get("pagination") or {}).get("hasMore"):
                break

    # Reconstruct recently closed rows on a first run; lifecycle state handles
    # them on subsequent runs. Canonical classification and the history-window
    # overlap filter below keep old or irrelevant results out of the timeline.
    for query in CLOSED_SEARCH_QUERIES:
        data = get_json(
            "/public-search",
            {
                "q": query,
                "events_status": "closed",
                "limit_per_type": 50,
                "page": 1,
                "keep_closed_markets": 1,
                "search_profiles": "false",
            },
        )
        for event in data.get("events") or []:
            add(event, f"closed-search:{query}")

    for tag_slug in tag_slugs:
        for offset in range(0, 2_000, 100):
            tagged = get_json(
                "/events",
                {
                    "tag_slug": tag_slug,
                    "related_tags": "true",
                    "active": "true",
                    "closed": "false",
                    "limit": 100,
                    "offset": offset,
                },
            )
            for event in tagged:
                add(event, f"tag:{tag_slug}")
            if len(tagged) < 100:
                break

    previous_max = int(previous_state.get("max_event_id") or 0)
    newest_id = previous_max
    cursor: str | None = None
    reached_watermark = previous_max == 0
    for page_number in range(INCREMENTAL_SCAN_PAGE_LIMIT):
        params: dict[str, Any] = {
            "active": "true",
            "closed": "false",
            "limit": 100,
            "order": "id",
            "ascending": "false",
        }
        if cursor:
            params["after_cursor"] = cursor
        data = get_json("/events/keyset", params)
        page_events = data.get("events") or []
        if page_number == 0:
            newest_id = max((numeric_event_id(event) for event in page_events), default=previous_max)
        for event in page_events:
            event_id = numeric_event_id(event)
            if previous_max and event_id <= previous_max:
                reached_watermark = True
                break
            if candidate_mentions_provider(event):
                add(event, "incremental")
        if reached_watermark or not page_events or not data.get("next_cursor"):
            break
        cursor = str(data["next_cursor"])
    if previous_max and not reached_watermark:
        warnings.append(
            f"Incremental scan did not reach event ID {previous_max} within {INCREMENTAL_SCAN_PAGE_LIMIT * 100} events; "
            "the watermark was not advanced."
        )

    forced = {slug_from_url(value) for value in extra_slugs}
    for slug in forced:
        add({"slug": slug}, "manual")
    for slug, record in (previous_state.get("events") or {}).items():
        add({"slug": slug}, "state")

    fetch_slugs = [
        slug
        for slug, summary in summaries.items()
        if candidate_mentions_provider(summary) or slug in forced or "state" in source_names.get(slug, set())
    ]

    def fetch(slug: str) -> tuple[str, dict[str, Any] | None]:
        records = get_json("/events", {"slug": slug})
        return slug, records[0] if records else None

    canonical: dict[str, dict[str, Any]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        for slug, event in executor.map(fetch, fetch_slugs):
            if event:
                canonical[slug] = event

    now = dt.datetime.now(dt.UTC).isoformat()
    previous_records = previous_state.get("events") or {}
    next_records: dict[str, dict[str, Any]] = {}
    accepted_events: list[dict[str, Any]] = []
    history_events: list[dict[str, Any]] = []
    decisions: list[DiscoveryDecision] = []
    for slug in sorted(set(fetch_slugs) | set(previous_records)):
        event = canonical.get(slug)
        previous = previous_records.get(slug) or {}
        if event is None:
            decisions.append(
                DiscoveryDecision(
                    str(previous.get("event_id") or ""),
                    slug,
                    str(previous.get("title") or slug),
                    previous.get("provider"),
                    str(previous.get("shape") or "unknown"),
                    "unavailable",
                    "changed" if previous else "new",
                    "The canonical event could not be fetched.",
                    tuple(sorted(source_names.get(slug, ()))),
                )
            )
            if previous:
                next_records[slug] = previous
            continue

        provider, shape, classification, reason = classify_event(event, forced=slug in forced)
        fingerprint = discovery_fingerprint(event)
        classification_changed = any(
            (
                previous.get("fingerprint") != fingerprint,
                previous.get("provider") != provider,
                previous.get("shape") != shape,
                previous.get("classification") != classification,
            )
        )
        change = "new" if not previous else ("changed" if classification_changed else "unchanged")
        decisions.append(
            DiscoveryDecision(
                str(event.get("id") or ""),
                slug,
                str(event.get("title") or slug),
                provider,
                shape,
                classification,
                change,
                reason,
                tuple(sorted(source_names.get(slug, ()))),
            )
        )
        next_records[slug] = {
            "event_id": str(event.get("id") or ""),
            "title": str(event.get("title") or slug),
            "provider": provider,
            "shape": shape,
            "classification": classification,
            "fingerprint": fingerprint,
            "first_seen": previous.get("first_seen") or now,
            "last_seen": now,
        }
        if classification == "accepted":
            accepted_events.append(event)
        if classification in {"accepted", "closed"}:
            history_events.append(event)

    next_state = {
        "version": DISCOVERY_STATE_VERSION,
        "max_event_id": newest_id if reached_watermark else previous_max,
        "updated_at": now,
        "events": next_records,
    }
    decisions.sort(key=lambda decision: (decision.classification != "accepted", decision.provider or "", decision.title))
    return DiscoveryResult(
        tuple(accepted_events), tuple(decisions), next_state, tuple(warnings), tuple(history_events)
    )


def discover_events(queries: Sequence[str], extra_slugs: Sequence[str]) -> list[dict[str, Any]]:
    """Compatibility wrapper returning accepted events without persistent state."""
    return list(discover_event_candidates(queries, extra_slugs).events)


def slug_from_url(value: str) -> str:
    parsed = urllib.parse.urlparse(value)
    return parsed.path.rstrip("/").split("/")[-1] if parsed.scheme else value


def parse_iso_datetime(value: Any) -> dt.datetime | None:
    if not value:
        return None
    try:
        parsed = dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=dt.UTC)
    except ValueError:
        return None


def yes_token_id(market: dict[str, Any]) -> str | None:
    try:
        outcomes = [str(value).lower() for value in unpack_json_list(market.get("outcomes"))]
        token_ids = [str(value) for value in unpack_json_list(market.get("clobTokenIds"))]
        return token_ids[outcomes.index("yes")]
    except (ValueError, IndexError, TypeError, json.JSONDecodeError):
        return None


def fetch_daily_price_histories(
    events: Sequence[dict[str, Any]], start: dt.date, end: dt.date
) -> dict[str, list[dict[str, float]]]:
    tokens = sorted(
        {
            token
            for event in events
            for market in event.get("markets") or []
            if (token := yes_token_id(market))
        }
    )
    if not tokens:
        return {}

    et = ZoneInfo("America/New_York")
    start_time = dt.datetime.combine(start, dt.time.min, et).astimezone(dt.UTC)
    end_time = dt.datetime.combine(end, dt.time.max, et).astimezone(dt.UTC)
    token_batches = [tokens[offset : offset + 20] for offset in range(0, len(tokens), 20)]
    time_chunks: list[tuple[dt.datetime, dt.datetime]] = []
    chunk_start = start_time
    while chunk_start < end_time:
        chunk_end = min(chunk_start + dt.timedelta(days=14), end_time)
        time_chunks.append((chunk_start, chunk_end))
        if chunk_end == end_time:
            break
        chunk_start = chunk_end

    def fetch(request: tuple[list[str], dt.datetime, dt.datetime]) -> dict[str, Any]:
        batch, period_start, period_end = request
        return post_clob_json(
            "/batch-prices-history",
            {
                "markets": batch,
                "start_ts": int(period_start.timestamp()),
                "end_ts": int(period_end.timestamp()),
                "fidelity": 1440,
            },
        )

    requests = [(batch, period_start, period_end) for batch in token_batches for period_start, period_end in time_chunks]
    merged: dict[str, dict[int, dict[str, float]]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        for response in executor.map(fetch, requests):
            for token, points in (response.get("history") or {}).items():
                token_points = merged.setdefault(str(token), {})
                for point in points or []:
                    token_points[int(point["t"])] = point
    histories = {
        token: [points[timestamp] for timestamp in sorted(points)] for token, points in merged.items()
    }
    return histories


def historical_price_at(
    history: Sequence[dict[str, float]], cutoff: dt.datetime, max_age_days: int = 3
) -> float | None:
    cutoff_timestamp = cutoff.timestamp()
    for point in reversed(history):
        timestamp = float(point["t"])
        if timestamp <= cutoff_timestamp:
            if cutoff_timestamp - timestamp > max_age_days * 86_400:
                return None
            return min(1.0, max(0.0, float(point["p"])))
    return None


def event_closed_at(event: dict[str, Any]) -> dt.datetime | None:
    direct = parse_iso_datetime(event.get("closedTime") or event.get("umaEndDate") or event.get("resolvedAt"))
    if direct:
        return direct
    market_times = [
        closed_at
        for market in event.get("markets") or []
        if (
            closed_at := parse_iso_datetime(
                market.get("closedTime") or market.get("umaEndDate") or market.get("resolvedAt")
            )
        )
    ]
    if market_times:
        return max(market_times)
    return parse_iso_datetime(event.get("updatedAt")) if event.get("closed") else None


def historical_events_at(
    events: Sequence[dict[str, Any]], histories: dict[str, list[dict[str, float]]], date: dt.date
) -> list[dict[str, Any]]:
    et = ZoneInfo("America/New_York")
    cutoff = dt.datetime.combine(date, dt.time.max, et).astimezone(dt.UTC)
    result: list[dict[str, Any]] = []
    for event in events:
        event_created = parse_iso_datetime(event.get("createdAt") or event.get("creationDate"))
        if event_created and event_created > cutoff:
            continue
        closed_at = event_closed_at(event)
        if event.get("closed") and closed_at and closed_at <= cutoff:
            continue
        historical_markets: list[dict[str, Any]] = []
        for market in event.get("markets") or []:
            market_created = parse_iso_datetime(market.get("createdAt") or market.get("creationDate"))
            if market_created and market_created > cutoff:
                continue
            token = yes_token_id(market)
            if not token or token not in histories:
                continue
            probability = historical_price_at(histories[token], cutoff)
            if probability is None:
                continue
            historical_market = dict(market)
            outcomes = [str(value).lower() for value in unpack_json_list(market.get("outcomes"))]
            prices = [1.0 - probability] * len(outcomes)
            prices[outcomes.index("yes")] = probability
            historical_market["outcomePrices"] = json.dumps([str(value) for value in prices])
            historical_market["lastTradePrice"] = probability
            closed_at = parse_iso_datetime(
                market.get("closedTime") or market.get("umaEndDate") or market.get("resolvedAt")
            )
            if closed_at and closed_at > cutoff:
                historical_market["closed"] = False
                historical_market["acceptingOrders"] = True
            for field in ("bestBid", "bestAsk", "spread"):
                historical_market.pop(field, None)
            historical_markets.append(historical_market)
        if historical_markets:
            historical_event = dict(event)
            historical_event["markets"] = historical_markets
            if event.get("closed"):
                historical_event["active"] = True
                historical_event["closed"] = False
            result.append(historical_event)
    return result


def provider_for(event: dict[str, Any]) -> str | None:
    title = str(event.get("title") or "").lower()
    questions = " ".join(str(market.get("question") or "") for market in event.get("markets") or []).lower()
    strong_text = f"{title} {questions}"
    matches = [name for name, terms in PROVIDER_TERMS.items() if any(term in strong_text for term in terms)]
    if len(matches) == 1:
        return matches[0]
    title_matches = [name for name, terms in PROVIDER_TERMS.items() if any(term in title for term in terms)]
    if len(title_matches) == 1:
        return title_matches[0]
    # Descriptions often mention competitors, so use them only as a unique fallback.
    text = f"{strong_text} {event.get('description', '')}".lower()
    fallback = [name for name, terms in PROVIDER_TERMS.items() if any(term in text for term in terms)]
    return fallback[0] if len(fallback) == 1 else None


def event_shape(event: dict[str, Any]) -> str:
    identity = release_identity(event)
    if identity:
        title = str(event.get("title") or "").lower()
        return "release-on" if "released on" in title else "release-by"
    yes_markets = [market for market in event.get("markets") or [] if yes_probability(market) is not None]
    dated = [market for market in yes_markets if market_date(market, event) is not None]
    if len(yes_markets) == 1 and dated:
        return "consumer-binary"
    if len(yes_markets) > 1 and len(dated) == len(yes_markets):
        return "consumer-deadlines"
    return "unsupported"


def classify_event(event: dict[str, Any], forced: bool = False) -> tuple[str | None, str, str, str]:
    provider = provider_for(event)
    shape = event_shape(event)
    title = str(event.get("title") or "").lower()
    questions = " ".join(str(market.get("question") or "") for market in event.get("markets") or []).lower()
    strong_text = f"{title} {questions}"
    if not event.get("markets"):
        return provider, shape, "unsupported", "The event has no markets."
    if provider is None:
        return None, shape, "rejected", "No supported provider could be identified from the title or market questions."
    if not forced and any(term in title for term in EXCLUDED_TITLE_TERMS):
        return provider, shape, "rejected", "The event is outside the consumer release/access scope."
    if not forced and not any(term in strong_text for term in CONSUMER_TERMS):
        return provider, shape, "rejected", "No consumer release, access, pricing, or retirement intent was found."
    if shape == "unsupported":
        return provider, shape, "unsupported", "The market shape is not yet supported."
    if event.get("closed") or not event.get("active"):
        return provider, shape, "closed", "The relevant event is no longer active."
    return provider, shape, "accepted", "Relevant supported consumer event."


def is_likely_relevant(event: dict[str, Any]) -> bool:
    title = str(event.get("title") or "").lower()
    questions = " ".join(str(market.get("question") or "") for market in event.get("markets") or []).lower()
    text = f"{title} {questions}"
    return (
        provider_for(event) is not None
        and any(term in text for term in CONSUMER_TERMS)
        and not any(term in title for term in EXCLUDED_TITLE_TERMS)
    )


def is_relevant(event: dict[str, Any]) -> bool:
    return classify_event(event)[2] == "accepted"


def unpack_json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        parsed = json.loads(value)
        return parsed if isinstance(parsed, list) else []
    return []


def yes_probability(market: dict[str, Any]) -> float | None:
    try:
        outcomes = [str(value).lower() for value in unpack_json_list(market.get("outcomes"))]
        prices = [float(value) for value in unpack_json_list(market.get("outcomePrices"))]
        return prices[outcomes.index("yes")]
    except (ValueError, IndexError, TypeError, json.JSONDecodeError):
        return None


def infer_year(text: str, event: dict[str, Any]) -> int:
    years = re.findall(r"\b(20\d{2})\b", text)
    if years:
        return int(years[-1])
    end_date = parse_iso_date(event.get("endDate"))
    return end_date.year if end_date else dt.date.today().year


def parse_human_date(text: str, event: dict[str, Any]) -> dt.date | None:
    match = DATE_RE.search(text)
    if not match:
        return None
    month, day, year = match.groups()
    try:
        return dt.date(int(year or infer_year(text, event)), MONTHS[month.lower()], int(day))
    except ValueError:
        return None


def market_date(market: dict[str, Any], event: dict[str, Any]) -> dt.date | None:
    # Questions usually include the year even when the shorter group title does not.
    return parse_human_date(str(market.get("question") or ""), event) or parse_human_date(
        str(market.get("groupItemTitle") or ""), event
    )


def parse_iso_date(value: Any) -> dt.date | None:
    if not value:
        return None
    try:
        return dt.datetime.fromisoformat(str(value).replace("Z", "+00:00")).date()
    except ValueError:
        return None


def money(value: float) -> str:
    if value >= 1_000_000:
        return f"${value / 1_000_000:.1f}m"
    if value >= 1_000:
        return f"${value / 1_000:.0f}k"
    return f"${value:.0f}"


def pct(value: float) -> str:
    return f"{100 * value:.1f}%" if value < 0.1 else f"{100 * value:.0f}%"


def pretty_date(value: dt.date) -> str:
    return f"{value:%b} {value.day}, {value.year}"


def market_today() -> dt.date:
    """Polymarket's model-release rules generally define deadlines in ET."""
    return dt.datetime.now(ZoneInfo("America/New_York")).date()


def isotonic_non_decreasing(values: Sequence[float]) -> list[float]:
    """Equal-weight pool-adjacent-violators regression."""
    blocks: list[list[float]] = []  # [mean, weight, start, end]
    for index, value in enumerate(values):
        blocks.append([value, 1.0, float(index), float(index)])
        while len(blocks) >= 2 and blocks[-2][0] > blocks[-1][0]:
            right = blocks.pop()
            left = blocks.pop()
            weight = left[1] + right[1]
            blocks.append(
                [
                    (left[0] * left[1] + right[0] * right[1]) / weight,
                    weight,
                    left[2],
                    right[3],
                ]
            )
    result = [0.0] * len(values)
    for mean, _weight, start, end in blocks:
        for index in range(int(start), int(end) + 1):
            result[index] = min(1.0, max(0.0, mean))
    return result


def quantile_from_cdf(points: Sequence[ProbabilityPoint], q: float) -> tuple[dt.date, str]:
    if points[0].probability >= q:
        return points[0].date, "upper"
    for previous, current in zip(points, points[1:]):
        if current.probability >= q:
            span = current.probability - previous.probability
            fraction = 1.0 if span <= 0 else (q - previous.probability) / span
            days = (current.date - previous.date).days
            return previous.date + dt.timedelta(days=round(days * fraction)), "interpolated"
    return points[-1].date, "lower"


def extract_cumulative_points(
    event: dict[str, Any], as_of: dt.date | None = None
) -> list[ProbabilityPoint]:
    """Extract and condition direct ``released by`` CDF quotes."""
    as_of = as_of or market_today()
    by_date: dict[dt.date, ProbabilityPoint] = {}
    for market in event.get("markets") or []:
        probability = yes_probability(market)
        date = market_date(market, event)
        question = str(market.get("question") or "").lower()
        if probability is None or date is None or " by " not in f" {question} ":
            continue
        point = ProbabilityPoint(
            date=date,
            probability=probability,
            label=str(market.get("groupItemTitle") or pretty_date(date)),
            volume=float(market.get("volume") or 0),
        )
        if date not in by_date or point.volume > by_date[date].volume:
            by_date[date] = point
    if not by_date:
        return []

    raw = sorted(by_date.values(), key=lambda point: point.date)
    # An expired zero-priced deadline tells us the release had not happened by
    # then. When the next quote is coarse (for example month-end), condition the
    # upcoming timeline on what is now known instead of interpolating mass into
    # the already elapsed gap.
    past = [point for point in raw if point.date < as_of]
    future = [point for point in raw if point.date >= as_of]
    if past and future and max(point.probability for point in past) <= 0.01:
        anchor_date = as_of - dt.timedelta(days=1)
        raw = [ProbabilityPoint(anchor_date, 0.0, "Known through yesterday", 0.0), *future]
    adjusted_probabilities = isotonic_non_decreasing([point.probability for point in raw])
    return [dataclasses.replace(point, probability=value) for point, value in zip(raw, adjusted_probabilities)]


def is_tail_bucket(market: dict[str, Any]) -> bool:
    text = f"{market.get('groupItemTitle', '')} {market.get('question', '')}".lower()
    return any(term in text for term in ("no release", "after ", "later than", "none of"))


def project_bounded_simplex(
    midpoints: Sequence[float], lower: Sequence[float], upper: Sequence[float], weights: Sequence[float]
) -> list[float]:
    """Reconcile mutually exclusive quotes to total 100% within bid/ask bounds."""
    if not midpoints:
        return []
    if sum(lower) > 1.0 + 1e-9 or sum(upper) < 1.0 - 1e-9:
        total = sum(midpoints)
        return [value / total for value in midpoints] if total else [1.0 / len(midpoints)] * len(midpoints)

    def values(lagrange: float) -> list[float]:
        return [
            min(hi, max(lo, midpoint - lagrange / max(weight, 1e-9)))
            for midpoint, lo, hi, weight in zip(midpoints, lower, upper, weights)
        ]

    low_lambda, high_lambda = -1_000_000.0, 1_000_000.0
    for _ in range(100):
        middle = (low_lambda + high_lambda) / 2
        if sum(values(middle)) > 1.0:
            low_lambda = middle
        else:
            high_lambda = middle
    return values((low_lambda + high_lambda) / 2)


def extract_exact_pdf(
    event: dict[str, Any], as_of: dt.date | None = None
) -> tuple[list[ProbabilityPoint], float]:
    """Return a coherent PDF and no-release tail from an exact-date event."""
    as_of = as_of or market_today()
    rows: list[tuple[dt.date | None, dict[str, Any], float]] = []
    for market in event.get("markets") or []:
        probability = yes_probability(market)
        if probability is None:
            continue
        if is_tail_bucket(market):
            rows.append((None, market, probability))
            continue
        date = market_date(market, event)
        question = str(market.get("question") or "").lower()
        if date and (" released on " in f" {question} " or "released on or prior" in question):
            rows.append((date, market, probability))
    if len(rows) < 2:
        return [], 0.0

    midpoints: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    weights: list[float] = []
    for date, market, midpoint in rows:
        if date is not None and date < as_of:
            midpoints.append(0.0)
            lower.append(0.0)
            upper.append(0.0)
            weights.append(1.0)
            continue
        bid = float(market.get("bestBid") or 0.0)
        ask = float(market.get("bestAsk") or 1.0)
        if ask < bid:
            bid, ask = ask, bid
        spread = max(ask - bid, 0.01)
        midpoints.append(min(ask, max(bid, midpoint)))
        lower.append(bid)
        upper.append(ask)
        weights.append(1.0 / (spread * spread))

    reconciled = project_bounded_simplex(midpoints, lower, upper, weights)
    dated: list[ProbabilityPoint] = []
    tail_probability = 0.0
    for (date, market, _midpoint), probability in zip(rows, reconciled):
        if date is None:
            tail_probability += probability
        else:
            dated.append(
                ProbabilityPoint(
                    date=date,
                    probability=probability,
                    label=str(market.get("groupItemTitle") or pretty_date(date)),
                    volume=float(market.get("volume") or 0),
                )
            )
    return sorted(dated, key=lambda point: point.date), tail_probability


RELEASE_VARIANT_RE = re.compile(r"\s+released\s+(?:by|on)\b.*$", re.IGNORECASE)


def release_identity(event: dict[str, Any]) -> tuple[str, str] | None:
    title = str(event.get("title") or "").strip()
    base = RELEASE_VARIANT_RE.sub("", title).strip(" .?…")
    if base == title.strip(" .?…"):
        return None
    key = re.sub(r"[^a-z0-9]+", " ", base.lower()).strip()
    key = re.sub(r"\b(?:the|model)\b", "", key)
    key = re.sub(r"\s+", " ", key).strip()
    return key, base


def merge_cumulative_points(events: Sequence[dict[str, Any]], as_of: dt.date) -> list[ProbabilityPoint]:
    candidates: dict[dt.date, ProbabilityPoint] = {}
    for event in events:
        for point in extract_cumulative_points(event, as_of):
            if point.date not in candidates or point.volume > candidates[point.date].volume:
                candidates[point.date] = point
    points = sorted(candidates.values(), key=lambda point: point.date)
    adjusted = isotonic_non_decreasing([point.probability for point in points])
    return [dataclasses.replace(point, probability=value) for point, value in zip(points, adjusted)]


def combine_cdf_and_pdf(
    anchors: Sequence[ProbabilityPoint], exact_pdf: Sequence[ProbabilityPoint]
) -> tuple[DistributionPoint, ...]:
    """Use exact-date shape inside cumulative-market probability anchors."""
    if not anchors and not exact_pdf:
        return ()
    if not anchors:
        cumulative = 0.0
        result = []
        for point in exact_pdf:
            cumulative += point.probability
            result.append(DistributionPoint(point.date, min(cumulative, 1.0), point.probability))
        return tuple(result)

    anchor_map = {point.date: point.probability for point in anchors}
    anchor_dates = sorted(anchor_map)
    pdf_map = {point.date: point.probability for point in exact_pdf}
    all_dates = sorted(set(anchor_dates) | {point.date for point in exact_pdf})
    cdf_by_date: dict[dt.date, float] = dict(anchor_map)

    # Exact-date buckets shape the mass between direct CDF anchors, while each
    # "by" quote remains an exact calibration point.
    for left, right in zip(anchor_dates, anchor_dates[1:]):
        intermediate = [date for date in all_dates if left < date <= right]
        if not intermediate:
            continue
        delta = max(0.0, anchor_map[right] - anchor_map[left])
        raw_mass = sum(pdf_map.get(date, 0.0) for date in intermediate)
        running = anchor_map[left]
        for date in intermediate:
            if date == right:
                cdf_by_date[date] = anchor_map[right]
            elif raw_mass > 0:
                running += delta * pdf_map.get(date, 0.0) / raw_mass
                cdf_by_date[date] = running
            else:
                fraction = (date - left).days / max(1, (right - left).days)
                cdf_by_date[date] = anchor_map[left] + delta * fraction

    # Exact dates before the first direct quote are calibrated to that quote.
    before = [date for date in all_dates if date < anchor_dates[0]]
    if before:
        first = anchor_dates[0]
        mass = sum(pdf_map.get(date, 0.0) for date in before + [first])
        running = 0.0
        for date in before:
            running += anchor_map[first] * pdf_map.get(date, 0.0) / mass if mass else 0.0
            cdf_by_date[date] = running

    after = [date for date in all_dates if date > anchor_dates[-1]]
    if after:
        last = anchor_dates[-1]
        remaining_exact_mass = sum(pdf_map.get(date, 0.0) for date in after)
        target = max(anchor_map[last], min(1.0, sum(pdf_map.values())))
        delta = target - anchor_map[last]
        running = anchor_map[last]
        for date in after:
            running += delta * pdf_map.get(date, 0.0) / remaining_exact_mass if remaining_exact_mass else 0.0
            cdf_by_date[date] = running

    ordered = sorted(cdf_by_date.items())
    cdf_values = isotonic_non_decreasing([value for _date, value in ordered])
    result: list[DistributionPoint] = []
    previous = 0.0
    for (date, _value), cdf in zip(ordered, cdf_values):
        cdf = min(1.0, max(previous, cdf))
        result.append(DistributionPoint(date, cdf, max(0.0, cdf - previous)))
        previous = cdf
    return tuple(result)


def source_for(event: dict[str, Any], label: str) -> MarketSource:
    return MarketSource(
        label=label,
        title=str(event.get("title") or "Untitled event"),
        url=POLYMARKET_EVENT_URL.format(event["slug"]),
        event_id=str(event.get("id") or ""),
    )


def combined_release_item(
    events: Sequence[dict[str, Any]], provider: str, as_of: dt.date | None = None
) -> TimelineItem | None:
    as_of = as_of or market_today()
    by_events = [event for event in events if "released by" in str(event.get("title") or "").lower()]
    on_events = [event for event in events if "released on" in str(event.get("title") or "").lower()]
    anchors = merge_cumulative_points(by_events, as_of)

    exact_pdf: list[ProbabilityPoint] = []
    if on_events:
        best_on_event = max(on_events, key=lambda event: float(event.get("volume") or 0))
        exact_pdf, _tail = extract_exact_pdf(best_on_event, as_of)
    distribution = combine_cdf_and_pdf(anchors, exact_pdf)
    if not distribution:
        return None

    primary = max(events, key=lambda event: float(event.get("volume") or 0))
    identity = release_identity(primary)
    title = identity[1] if identity else str(primary.get("title") or "Untitled release")
    horizon = distribution[-1]
    median_date: dt.date | None = None
    median_kind = "lower"
    if horizon.cdf >= 0.5:
        if exact_pdf and not anchors:
            median_date = next(point.date for point in distribution if point.cdf >= 0.5)
            median_kind = "exact"
        else:
            probability_points = [ProbabilityPoint(point.date, point.cdf, "", 0.0) for point in distribution]
            median_date, median_kind = quantile_from_cdf(probability_points, 0.5)

    if median_date is None:
        when = "Median not established"
        estimate = f"Only {pct(horizon.cdf)} probability by {pretty_date(horizon.date)}"
        detail = "Later dates are not priced by these markets, so no median release date is inferred."
        sort_date = horizon.date
    else:
        when = f"by {pretty_date(median_date)}" if median_kind == "upper" else f"around {pretty_date(median_date)}"
        estimate = "median release"
        detail = f"The combined distribution reaches {pct(horizon.cdf)} by {pretty_date(horizon.date)}."
        sort_date = median_date

    sources = tuple(
        [source_for(event, "By-deadline market") for event in by_events]
        + [source_for(event, "Exact-date market") for event in on_events]
    )
    return TimelineItem(
        provider=provider,
        title=title,
        kind="release forecast" + (" (combined by/on markets)" if by_events and on_events else ""),
        sort_date=sort_date,
        when=when,
        estimate=estimate,
        detail=detail,
        volume=sum(float(event.get("volume") or 0) for event in events),
        url=POLYMARKET_EVENT_URL.format(primary["slug"]),
        event_id=str(primary.get("id") or ""),
        distribution=distribution,
        sources=sources,
        median_date=median_date,
    )


def cumulative_item(event: dict[str, Any], provider: str, as_of: dt.date | None = None) -> TimelineItem | None:
    return combined_release_item([event], provider, as_of)


def exact_date_item(event: dict[str, Any], provider: str, as_of: dt.date | None = None) -> TimelineItem | None:
    return combined_release_item([event], provider, as_of)


def consumer_change_title(event: dict[str, Any]) -> str:
    title = str(event.get("title") or "Untitled consumer change").strip()
    return re.sub(r"\s+by(?:\.{3}|…)?\?\s*$", "", title, flags=re.IGNORECASE).strip()


def resolved_yes_market(event: dict[str, Any]) -> tuple[dict[str, Any], dt.datetime] | None:
    resolved: list[tuple[dict[str, Any], dt.datetime]] = []
    for market in event.get("markets") or []:
        probability = yes_probability(market)
        if not market.get("closed") or probability is None or probability < 0.99:
            continue
        closed_at = parse_iso_datetime(
            market.get("closedTime") or market.get("umaEndDate") or market.get("resolvedAt") or market.get("updatedAt")
        )
        if closed_at:
            resolved.append((market, closed_at))
    return min(resolved, key=lambda row: row[1]) if resolved else None


def consumer_change_item(
    event: dict[str, Any], provider: str, as_of: dt.date | None = None
) -> TimelineItem | None:
    as_of = as_of or market_today()
    markets = [market for market in event.get("markets") or [] if yes_probability(market) is not None]
    if not markets:
        return None
    title = consumer_change_title(event)
    resolved = resolved_yes_market(event)
    if resolved:
        market, closed_at = resolved
        resolved_date = closed_at.astimezone(ZoneInfo("America/New_York")).date()
        return TimelineItem(
            provider=provider,
            title=title,
            kind="consumer change (resolved)",
            sort_date=resolved_date,
            when=f"resolved {pretty_date(resolved_date)}",
            estimate="Yes — change occurred",
            detail=(
                f'The market "{market.get("question") or title}" resolved Yes. '
                "Later cumulative deadlines no longer represent independent future outcomes."
            ),
            volume=float(event.get("volume") or market.get("volume") or 0),
            url=POLYMARKET_EVENT_URL.format(event["slug"]),
            event_id=str(event.get("id") or ""),
        )

    if len(markets) == 1:
        market = markets[0]
        probability = yes_probability(market)
        assert probability is not None
        deadline = (
            market_date(market, event)
            or parse_iso_date(market.get("endDate"))
            or parse_iso_date(event.get("endDate"))
        )
        if deadline is None:
            return None
        return TimelineItem(
            provider=provider,
            title=title,
            kind="consumer change (binary)",
            sort_date=deadline,
            when=f"by {pretty_date(deadline)}",
            estimate=f"Yes {pct(probability)} / No {pct(1 - probability)}",
            detail="This is an outcome probability, not an expected date; check the linked resolution rules for what Yes means.",
            volume=float(event.get("volume") or market.get("volume") or 0),
            url=POLYMARKET_EVENT_URL.format(event["slug"]),
            event_id=str(event.get("id") or ""),
        )

    anchors = extract_cumulative_points(event, as_of)
    distribution = combine_cdf_and_pdf(anchors, [])
    if not distribution:
        return None
    horizon = distribution[-1]
    median_date: dt.date | None = None
    median_kind = "lower"
    if horizon.cdf >= 0.5:
        median_date, median_kind = quantile_from_cdf(
            [ProbabilityPoint(point.date, point.cdf, "", 0.0) for point in distribution], 0.5
        )
    if median_date is None:
        when = "Median not established"
        estimate = f"Only {pct(horizon.cdf)} probability by {pretty_date(horizon.date)}"
        detail = "Later dates are not priced by this market, so no median date for the change is inferred."
        sort_date = horizon.date
    else:
        when = f"by {pretty_date(median_date)}" if median_kind == "upper" else f"around {pretty_date(median_date)}"
        estimate = "median date for change"
        detail = f"The cumulative probability reaches {pct(horizon.cdf)} by {pretty_date(horizon.date)}."
        sort_date = median_date
    return TimelineItem(
        provider=provider,
        title=title,
        kind="consumer change forecast (multi-deadline)",
        sort_date=sort_date,
        when=when,
        estimate=estimate,
        detail=detail,
        volume=float(event.get("volume") or 0),
        url=POLYMARKET_EVENT_URL.format(event["slug"]),
        event_id=str(event.get("id") or ""),
        distribution=distribution,
        sources=(source_for(event, "Deadline market"),),
        median_date=median_date,
    )


def binary_item(event: dict[str, Any], provider: str) -> TimelineItem | None:
    """Backward-compatible name for consumer-change rendering."""
    return consumer_change_item(event, provider)


def event_to_item(event: dict[str, Any], as_of: dt.date | None = None) -> TimelineItem | None:
    provider = provider_for(event)
    if provider is None:
        return None
    if release_identity(event):
        return combined_release_item([event], provider, as_of)
    return consumer_change_item(event, provider, as_of)


def build_timeline(
    events: Iterable[dict[str, Any]], min_volume: float = 0.0, as_of: dt.date | None = None
) -> list[TimelineItem]:
    release_groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    other_events: list[dict[str, Any]] = []
    for event in events:
        provider = provider_for(event)
        identity = release_identity(event)
        if provider and identity:
            release_groups.setdefault((provider, identity[0]), []).append(event)
        else:
            other_events.append(event)

    items: list[TimelineItem] = []
    for (provider, _identity), grouped_events in release_groups.items():
        if item := combined_release_item(grouped_events, provider, as_of):
            items.append(item)
    items.extend(item for event in other_events if (item := event_to_item(event, as_of)))
    items = [item for item in items if item.volume >= min_volume]
    return sorted(items, key=lambda item: (item.sort_date, item.provider, -item.volume))


def event_overlaps_history(event: dict[str, Any], start: dt.date, end: dt.date) -> bool:
    created_at = parse_iso_datetime(event.get("createdAt") or event.get("creationDate"))
    closed_at = event_closed_at(event)
    return (created_at is None or created_at.date() <= end) and (closed_at is None or closed_at.date() >= start)


def build_daily_history(
    events: Sequence[dict[str, Any]],
    current_items: Sequence[TimelineItem],
    days: int,
    min_volume: float = 0.0,
) -> list[HistorySnapshot]:
    end = market_today()
    if days <= 1:
        return [HistorySnapshot(end, tuple(current_items))]
    start = end - dt.timedelta(days=days - 1)
    events = [event for event in events if event_overlaps_history(event, start, end)]
    histories = fetch_daily_price_histories(events, start, end)
    snapshots: list[HistorySnapshot] = []
    date = start
    while date < end:
        historical_events = historical_events_at(events, histories, date)
        historical_items = build_timeline(historical_events, min_volume, as_of=date)
        snapshots.append(HistorySnapshot(date, tuple(historical_items)))
        date += dt.timedelta(days=1)
    snapshots.append(HistorySnapshot(end, tuple(current_items)))
    return snapshots


def markdown_escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def render_markdown(items: Sequence[TimelineItem], generated_at: dt.datetime) -> str:
    lines = [
        f"# {SITE_TITLE}",
        "",
        f"Generated {generated_at.astimezone().strftime('%Y-%m-%d %H:%M %Z')} from live Polymarket quotes.",
        "",
        "| When | Provider | Market-implied signal | Market | Volume |",
        "|---|---|---|---|---:|",
    ]
    for item in items:
        signal = f"{item.estimate}. {item.detail}"
        if item.sources:
            link = "; ".join(f"[{source.label}]({source.url})" for source in item.sources)
        else:
            link = f"[{markdown_escape(item.title)}]({item.url})"
        lines.append(
            f"| {markdown_escape(item.when)} | {item.provider} | {markdown_escape(signal)} | {link} | {money(item.volume)} |"
        )
    if not items:
        lines.append("| — | — | No matching active markets found. | — | — |")
    lines += [
        "",
        "Median estimates use market prices as a cumulative distribution and linearly interpolate between quoted deadlines. "
        "Exclusive date markets are normalized because independently quoted prices need not sum to exactly 100%. "
        "Prices are forecasts, not facts, and can be noisy or internally inconsistent.",
    ]
    return "\n".join(lines)


def render_discovery_report(result: DiscoveryResult, generated_at: dt.datetime) -> str:
    counts: dict[str, int] = {}
    for decision in result.decisions:
        counts[decision.classification] = counts.get(decision.classification, 0) + 1
    summary = ", ".join(f"{name} {count}" for name, count in sorted(counts.items())) or "no candidates"
    lines = [
        "# Polymarket discovery audit",
        "",
        f"Generated {generated_at.astimezone().strftime('%Y-%m-%d %H:%M %Z')}. {summary}.",
        "",
    ]
    for warning in result.warnings:
        lines.append(f"> Warning: {warning}")
    if result.warnings:
        lines.append("")
    lines += [
        "| Change | Classification | Provider | Shape | Event | Reason | Sources |",
        "|---|---|---|---|---|---|---|",
    ]
    priority = {"new": 0, "changed": 1, "unchanged": 2}
    for decision in sorted(
        result.decisions,
        key=lambda row: (priority.get(row.change, 9), row.classification, row.provider or "", row.title),
    ):
        event_link = f"[{markdown_escape(decision.title)}]({POLYMARKET_EVENT_URL.format(decision.slug)})"
        lines.append(
            "| "
            + " | ".join(
                (
                    decision.change,
                    decision.classification,
                    decision.provider or "—",
                    decision.shape,
                    event_link,
                    markdown_escape(decision.reason),
                    markdown_escape(", ".join(decision.sources) or "—"),
                )
            )
            + " |"
        )
    if not result.decisions:
        lines.append("| — | — | — | — | No candidates found | — | — |")
    return "\n".join(lines)


def daily_pdf_rates(points: Sequence[DistributionPoint]) -> list[float]:
    """Convert probability mass per interval into probability density per day."""
    rates: list[float] = []
    previous_date: dt.date | None = None
    for point in points:
        days = max(1, (point.date - previous_date).days) if previous_date else 1
        rates.append(point.pdf / days)
        previous_date = point.date
    return rates


def month_boundaries(start: dt.date, end: dt.date) -> list[dt.date]:
    """Return first-of-month dates strictly inside a date range."""
    if start >= end:
        return []
    if start.month == 12:
        current = dt.date(start.year + 1, 1, 1)
    else:
        current = dt.date(start.year, start.month + 1, 1)
    result: list[dt.date] = []
    while current < end:
        result.append(current)
        if current.month == 12:
            current = dt.date(current.year + 1, 1, 1)
        else:
            current = dt.date(current.year, current.month + 1, 1)
    return result


def axis_tick_dates(
    start: dt.date,
    end: dt.date,
    *,
    max_ticks: int = 8,
    minimum_separation: float = 0.07,
) -> list[dt.date]:
    """Choose calendar-aligned labels without crowding nearby endpoints."""
    if start >= end:
        return [start]

    boundaries = month_boundaries(start, end)
    if not boundaries:
        return [start, end]

    max_ticks = max(2, max_ticks)
    if len(boundaries) > max_ticks:
        last_index = len(boundaries) - 1
        boundaries = list(
            dict.fromkeys(
                boundaries[round(index * last_index / (max_ticks - 1))]
                for index in range(max_ticks)
            )
        )

    selected = list(boundaries)
    span_days = (end - start).days
    for endpoint in (start, end):
        if len(selected) >= max_ticks:
            break
        if all(abs((endpoint - tick).days) / span_days >= minimum_separation for tick in selected):
            selected.append(endpoint)

    if len(selected) == 1:
        selected.append(max((start, end), key=lambda endpoint: abs((endpoint - selected[0]).days)))
    return sorted(set(selected))


def render_probability_chart(item: TimelineItem, as_of: dt.date | None = None) -> str:
    """Render compact, accessible PDF/CDF small multiples as inline SVG."""
    points = item.distribution
    if not points:
        return ""
    is_change = item.kind.startswith("consumer change")
    outcome = "change" if is_change else "release"
    cumulative_label = "change occurred by date" if is_change else "released by date"
    pdf_label = "change probability rate per day" if is_change else "probability rate per day"

    width, height = 760, 270
    left, right = 58.0, 742.0
    cdf_top, cdf_bottom = 26.0, 126.0
    pdf_top, pdf_bottom = 172.0, 232.0
    axis_start = min(points[0].date, as_of) if as_of else points[0].date
    axis_end = max(points[-1].date, as_of) if as_of else points[-1].date
    first_ordinal = axis_start.toordinal()
    last_ordinal = axis_end.toordinal()
    span = max(1, last_ordinal - first_ordinal)

    def x(date: dt.date) -> float:
        return left + (date.toordinal() - first_ordinal) / span * (right - left)

    def cdf_y(probability: float) -> float:
        return cdf_bottom - probability * (cdf_bottom - cdf_top)

    pdf_rates = daily_pdf_rates(points)
    max_pdf_rate = max(pdf_rates, default=0.0) or 1.0

    def pdf_y(rate: float) -> float:
        return pdf_bottom - rate / max_pdf_rate * (pdf_bottom - pdf_top)

    esc = lambda value: html.escape(str(value), quote=True)
    line_points = " ".join(f"{x(point.date):.1f},{cdf_y(point.cdf):.1f}" for point in points)
    marks: list[str] = []
    previous_x = x(points[0].date)
    previous_date: dt.date | None = None
    for point, daily_rate in zip(points, pdf_rates):
        current_x = x(point.date)
        bar_left = previous_x + 1.5
        bar_width = max(3.0, current_x - previous_x - 3.0)
        bar_top = pdf_y(daily_rate)
        interval_days = max(1, (point.date - previous_date).days) if previous_date else 1
        tooltip = (
            f"{pretty_date(point.date)} — {cumulative_label}: {pct(point.cdf)}; "
            f"{outcome} rate: {pct(daily_rate)} per day over {interval_days} day{'s' if interval_days != 1 else ''}"
        )
        marks.append(
            f'<rect class="pdf-bar" x="{bar_left:.1f}" y="{bar_top:.1f}" width="{bar_width:.1f}" '
            f'height="{pdf_bottom - bar_top:.1f}" data-chart-tooltip="{esc(tooltip)}" aria-label="{esc(tooltip)}" />'
        )
        marks.append(
            f'<circle class="cdf-point" cx="{current_x:.1f}" cy="{cdf_y(point.cdf):.1f}" r="4" '
            f'data-chart-tooltip="{esc(tooltip)}" aria-label="{esc(tooltip)}" />'
        )
        previous_x = current_x
        previous_date = point.date

    boundaries = month_boundaries(axis_start, axis_end)
    month_lines = "".join(
        f'<line class="month-grid" x1="{x(date):.1f}" y1="{cdf_top}" x2="{x(date):.1f}" y2="{pdf_bottom}" />'
        for date in boundaries
    )
    tick_dates = axis_tick_dates(axis_start, axis_end)
    ticks: list[str] = []
    for date in tick_dates:
        tick_x = x(date)
        label = date.strftime("%b %d").replace(" 0", " ")
        ticks.append(
            f'<line class="axis-tick" x1="{tick_x:.1f}" y1="{pdf_bottom}" x2="{tick_x:.1f}" y2="{pdf_bottom + 5}" />'
            f'<text class="axis-label" x="{tick_x:.1f}" y="{pdf_bottom + 22}" text-anchor="middle">{esc(label)}</text>'
        )

    median_mark = ""
    if item.median_date:
        median_x = x(item.median_date)
        median_mark = (
            f'<line class="median-line" x1="{median_x:.1f}" y1="{cdf_top}" x2="{median_x:.1f}" y2="{pdf_bottom}" />'
            f'<text class="median-label" x="{median_x + 5:.1f}" y="{cdf_top + 11}">median</text>'
        )

    as_of_mark = ""
    if as_of:
        as_of_x = x(as_of)
        as_of_text = f"Forecast as of {pretty_date(as_of)}"
        label_anchor = "end" if as_of_x > right - 90 else "start"
        label_x = as_of_x - 5 if label_anchor == "end" else as_of_x + 5
        as_of_mark = (
            f'<line class="as-of-line" x1="{as_of_x:.1f}" y1="{cdf_top}" x2="{as_of_x:.1f}" y2="{pdf_bottom}" '
            f'data-chart-tooltip="{esc(as_of_text)}" aria-label="{esc(as_of_text)}" />'
            f'<text class="as-of-label" x="{label_x:.1f}" y="{cdf_top + 24}" text-anchor="{label_anchor}" '
            f'data-chart-tooltip="{esc(as_of_text)}">as of {esc(as_of.strftime("%b %d").replace(" 0", " "))}</text>'
        )

    summary = ", ".join(f"{pretty_date(point.date)} {pct(point.cdf)} cumulative" for point in points)
    as_of_description = f" Forecast as of {pretty_date(as_of)}." if as_of else ""
    return f"""<svg class="prob-chart" viewBox="0 0 {width} {height}" role="img" aria-label="{outcome.title()} probability PDF and CDF by date.{esc(as_of_description)}">
              <title>{esc(item.title)} {outcome} probability</title>
              <desc>{esc(summary)}.{esc(as_of_description)}</desc>
              <text class="plot-label" x="{left}" y="15">CDF · probability {esc(cumulative_label)}</text>
              {month_lines}
              <line class="grid-line" x1="{left}" y1="{cdf_top}" x2="{right}" y2="{cdf_top}" />
              <line class="grid-line threshold" x1="{left}" y1="{cdf_y(0.5)}" x2="{right}" y2="{cdf_y(0.5)}" />
              <line class="grid-line" x1="{left}" y1="{cdf_bottom}" x2="{right}" y2="{cdf_bottom}" />
              <text class="axis-label" x="{left - 8}" y="{cdf_top + 4}" text-anchor="end">100%</text>
              <text class="axis-label" x="{left - 8}" y="{cdf_y(0.5) + 4}" text-anchor="end">50%</text>
              <text class="axis-label" x="{left - 8}" y="{cdf_bottom + 4}" text-anchor="end">0%</text>
              <polyline class="cdf-line" points="{line_points}" />
              {''.join(marks)}
              {median_mark}
              {as_of_mark}
              <text class="plot-label" x="{left}" y="161">PDF · {esc(pdf_label)}</text>
              <text class="axis-label" x="{left - 8}" y="{pdf_top + 4}" text-anchor="end">{esc(pct(max_pdf_rate))}/day</text>
              <text class="axis-label" x="{left - 8}" y="{pdf_bottom + 4}" text-anchor="end">0%</text>
              <line class="grid-line" x1="{left}" y1="{pdf_bottom}" x2="{right}" y2="{pdf_bottom}" />
              {''.join(ticks)}
            </svg>"""


def summary_date_signal(item: TimelineItem) -> tuple[str, dt.date, str]:
    """Return marker kind, plotted date, and accessible table label."""
    if item.median_date:
        return "median", item.median_date, f"Median {pretty_date(item.median_date)}"
    if item.distribution:
        horizon = item.distribution[-1]
        return (
            "horizon",
            horizon.date,
            f"No median; {pct(horizon.cdf)} probability by {pretty_date(horizon.date)}",
        )
    return "deadline", item.sort_date, f"Decision {item.when}; {item.estimate}"


def item_quantile(item: TimelineItem, quantile: float) -> tuple[dt.date, str] | None:
    """Return a release quantile when it is bounded by the market horizon."""
    if not item.distribution or item.distribution[-1].cdf < quantile:
        return None
    points = [ProbabilityPoint(point.date, point.cdf, "", 0.0) for point in item.distribution]
    return quantile_from_cdf(points, quantile)


def timeline_item_key(item: TimelineItem) -> str:
    title = re.sub(r"[^a-z0-9]+", " ", item.title.lower()).strip()
    return f"{item.provider.lower()}::{title}"


def timeline_item_anchor(item: TimelineItem) -> str:
    """Return the stable detail-card fragment for an overview item."""
    slug = re.sub(r"[^a-z0-9]+", "-", timeline_item_key(item)).strip("-")
    return f"detail-{slug}"


def summary_roster(history: Sequence[HistorySnapshot]) -> tuple[TimelineItem, ...]:
    """Return one latest representative per item in a stable display order."""
    representatives: dict[str, TimelineItem] = {}
    for snapshot in history:
        for item in snapshot.items:
            representatives[timeline_item_key(item)] = item
    return tuple(
        sorted(
            representatives.values(),
            key=lambda item: (summary_date_signal(item)[1], item.provider, item.title),
        )
    )


def render_summary_card(
    items: Sequence[TimelineItem],
    axis_range: tuple[dt.date, dt.date] | None = None,
    roster: Sequence[TimelineItem] | None = None,
    as_of: dt.date | None = None,
) -> str:
    """Render a shared date plot with a stable roster across history snapshots."""
    esc = lambda value: html.escape(str(value), quote=True)
    current_by_key = {timeline_item_key(item): item for item in items}
    if roster is None:
        roster = tuple(
            sorted(items, key=lambda item: (summary_date_signal(item)[1], item.provider, item.title))
        )
    else:
        roster = tuple(roster)
    if not roster:
        return """<section class="summary-card">
      <h2>Overview</h2>
      <p class="detail">No matching active markets found.</p>
    </section>"""

    table_rows: list[str] = []
    for representative in roster:
        key = timeline_item_key(representative)
        item = current_by_key.get(key)
        signal_text = summary_date_signal(item)[2] if item else "Not available on this date"
        row_class = "summary-table-row" + (
            " unavailable" if item is None or item.median_date is None else ""
        )
        title_html = (
            f'<a href="#{esc(timeline_item_anchor(item))}">{esc(representative.title)}</a>'
            if item
            else f'<span>{esc(representative.title)}</span>'
        )
        table_rows.append(
            f"""            <tr class="{row_class}" data-summary-key="{esc(key)}">
              <td>{title_html}</td>
              <td>{esc(representative.provider)}</td>
              <td>{esc(signal_text)}</td>
            </tr>"""
        )

    range_dates: list[dt.date] = []
    for representative in roster:
        item = current_by_key.get(timeline_item_key(representative)) or representative
        if item.median_date:
            q25 = item_quantile(item, 0.25)
            q75 = item_quantile(item, 0.75)
            horizon = item.distribution[-1].date if item.distribution else item.median_date
            iqr_start = q25[0] if q25 else item.median_date
            iqr_end = q75[0] if q75 else horizon
            range_dates.extend((iqr_start, item.median_date, iqr_end))
    if range_dates and as_of:
        range_dates.append(as_of)

    if axis_range or range_dates:
        if axis_range:
            min_date, max_date = axis_range
        else:
            min_date = min(range_dates) - dt.timedelta(days=2)
            max_date = max(range_dates) + dt.timedelta(days=2)
        span = max(1, (max_date - min_date).days)

        def position(date: dt.date) -> float:
            return max(0.0, min(100.0, (date - min_date).days / span * 100))

        boundaries = month_boundaries(min_date, max_date)
        month_grid = "".join(
            f'<i class="summary-month-grid" style="left: {position(date):.2f}%" aria-hidden="true"></i>'
            for date in boundaries
        )
        plot_rows: list[str] = []
        for representative in roster:
            key = timeline_item_key(representative)
            item = current_by_key.get(key)
            provider_class = representative.provider.lower().replace(" ", "-")
            if item is None or item.median_date is None:
                reason = "Not available on this date" if item is None else summary_date_signal(item)[2]
                title_html = (
                    f'<a class="summary-plot-label" href="#{esc(timeline_item_anchor(item))}" title="{esc(item.title)}">{esc(representative.title)}</a>'
                    if item
                    else f'<span class="summary-plot-label" title="{esc(representative.title)}">{esc(representative.title)}</span>'
                )
                plot_rows.append(
                    f"""        <div class="summary-plot-row {esc(provider_class)} unavailable" data-summary-key="{esc(key)}" aria-label="{esc(representative.title)}: {esc(reason)}">
          {title_html}
          <span class="summary-track" aria-hidden="true">{month_grid}</span>
        </div>"""
                )
                continue

            date = item.median_date
            signal_text = summary_date_signal(item)[2]
            q25 = item_quantile(item, 0.25)
            q75 = item_quantile(item, 0.75)
            horizon = item.distribution[-1].date if item.distribution else date
            iqr_start = q25[0] if q25 else date
            iqr_end = q75[0] if q75 else horizon
            q25_text = (
                ("on or before " if q25[1] == "upper" else "") + pretty_date(q25[0]) if q25 else "not bounded"
            )
            q75_text = pretty_date(q75[0]) if q75 else f"beyond the {pretty_date(horizon)} market horizon"
            tooltip = (
                f"{item.title} — {item.provider}. {signal_text}. "
                f"25th percentile: {q25_text}; 75th percentile: {q75_text}. {item.detail}"
            ).strip()
            iqr_classes = ["summary-iqr"]
            if q25 is None or q25[1] == "upper":
                iqr_classes.append("open-left")
            if q75 is None or q75[1] == "lower":
                iqr_classes.append("open-right")
            iqr_left = position(iqr_start)
            iqr_width = max(0.0, position(iqr_end) - iqr_left)
            plot_rows.append(
                f"""        <div class="summary-plot-row {esc(provider_class)}" data-summary-key="{esc(key)}" aria-label="{esc(item.title)}: {esc(signal_text)}">
          <a class="summary-plot-label" href="#{esc(timeline_item_anchor(item))}" title="{esc(item.title)}">{esc(representative.title)}</a>
          <span class="summary-track" aria-hidden="true">
            {month_grid}
            <span class="{' '.join(iqr_classes)}" style="left: {iqr_left:.2f}%; width: {iqr_width:.2f}%"></span>
            <span class="summary-dot median" style="left: {position(date):.2f}%" data-chart-tooltip="{esc(tooltip)}"></span>
          </span>
        </div>"""
            )

        ticks: list[str] = []
        tick_dates = axis_tick_dates(min_date, max_date)
        for date in tick_dates:
            transform = "0" if date == min_date else ("-100%" if date == max_date else "-50%")
            ticks.append(
                f'<span class="summary-tick" style="left: {position(date):.2f}%; transform: translateX({transform})">'
                f'{esc(date.strftime("%b %d").replace(" 0", " "))}</span>'
            )
        as_of_overlay = ""
        if as_of:
            as_of_text = f"Forecast as of {pretty_date(as_of)}"
            as_of_overlay = f"""        <span class="summary-as-of-overlay">
          <span aria-hidden="true"></span>
          <span class="summary-as-of-area"><i class="summary-as-of" role="img" style="left: {position(as_of):.2f}%" data-chart-tooltip="{esc(as_of_text)}" aria-label="{esc(as_of_text)}"></i></span>
        </span>"""
        plot = f"""      <div class="summary-plot" role="group" aria-label="Median forecast dates on one date axis">
{chr(10).join(plot_rows)}
{as_of_overlay}
        <div class="summary-axis">
          <span></span>
          <span class="summary-axis-track">{''.join(ticks)}</span>
        </div>
      </div>"""
    else:
        plot = '<p class="detail">No forecast median is available in this history window.</p>'

    return f"""<section class="summary-card">
      <div class="summary-heading">
        <h2>Overview</h2>
        <div class="overview-filter" role="group" aria-label="Overview market visibility">
          <button type="button" data-overview-filter="all" aria-pressed="false">All markets</button>
          <button type="button" data-overview-filter="active" aria-pressed="true">Active only</button>
        </div>
      </div>
      <h3 class="summary-plot-title">Median forecast dates</h3>
      <p class="summary-plot-note">Circle: median · Solid line: 25–75% IQR · Vertical cursor: forecast date · Arrow: percentile beyond priced horizon · Grey: no median or unavailable on selected date</p>
{plot}
      <div class="summary-table-wrap">
        <table class="summary-table">
          <thead><tr><th scope="col">Model or change</th><th scope="col">Provider</th><th scope="col">Date signal</th></tr></thead>
          <tbody>
{chr(10).join(table_rows)}
          </tbody>
        </table>
      </div>
    </section>"""


def render_timeline_cards(items: Sequence[TimelineItem], as_of: dt.date | None = None) -> str:
    esc = lambda value: html.escape(str(value), quote=True)
    cards = []
    for item in items:
        provider_class = item.provider.lower().replace(" ", "-")
        anchor = timeline_item_anchor(item)
        if item.sources:
            heading = f"<h2>{esc(item.title)}</h2>"
            source_links = " · ".join(
                f'<a href="{esc(source.url)}">{esc(source.label)}</a>' for source in item.sources
            )
            sources = f'<p class="sources">Markets: {source_links}</p>'
        else:
            heading = f'<h2><a href="{esc(item.url)}">{esc(item.title)}</a></h2>'
            sources = ""
        chart = render_probability_chart(item, as_of=as_of)
        cards.append(
            f"""        <li class="timeline-item {esc(provider_class)}" id="{esc(anchor)}">
          <div class="marker" aria-hidden="true"></div>
          <article>
            <header class="card-header">
              <div>
                <time datetime="{item.sort_date.isoformat()}">{esc(item.when)}</time>
                <span class="provider">{esc(item.provider)}</span>
              </div>
              <span class="volume" title="Total market volume">{esc(money(item.volume))} volume</span>
            </header>
            {heading}
            <p class="estimate">{esc(item.estimate)}</p>
            <p class="detail">{esc(item.detail)}</p>
            {chart}
            {sources}
            <p class="kind">{esc(item.kind)}</p>
          </article>
        </li>"""
        )

    if not cards:
        cards.append(
            """        <li class="timeline-item empty">
          <div class="marker" aria-hidden="true"></div>
          <article><h2>No matching active markets found</h2></article>
        </li>"""
        )
    return "\n".join(cards)


def summary_axis_range(history: Sequence[HistorySnapshot]) -> tuple[dt.date, dt.date] | None:
    dates: list[dt.date] = []
    snapshot_dates: list[dt.date] = []
    for snapshot in history:
        snapshot_dates.append(snapshot.date)
        for item in snapshot.items:
            if not item.median_date:
                continue
            q25 = item_quantile(item, 0.25)
            q75 = item_quantile(item, 0.75)
            horizon = item.distribution[-1].date if item.distribution else item.median_date
            dates.extend((q25[0] if q25 else item.median_date, item.median_date, q75[0] if q75 else horizon))
    if not dates:
        return None
    dates.extend(snapshot_dates)
    return min(dates) - dt.timedelta(days=2), max(dates) + dt.timedelta(days=2)


def render_html(
    items: Sequence[TimelineItem],
    generated_at: dt.datetime,
    history: Sequence[HistorySnapshot] = (),
    site_url: str | None = None,
    source_url: str | None = None,
) -> str:
    """Render a complete, dependency-free HTML document."""

    def esc(value: Any) -> str:
        return html.escape(str(value), quote=True)

    history = tuple(history)
    axis_range = summary_axis_range(history) if history else None
    roster = summary_roster(history) if history else None
    current_as_of = (
        history[-1].date
        if history
        else generated_at.astimezone(ZoneInfo("America/New_York")).date()
    )
    cards_html = render_timeline_cards(items, as_of=current_as_of)

    generated_text = generated_at.astimezone().strftime("%Y-%m-%d %H:%M %Z")
    summary_card = render_summary_card(items, axis_range=axis_range, roster=roster, as_of=current_as_of)
    history_controls = ""
    history_data = ""
    generated_source = "live Polymarket quotes"
    canonical_url = f"{site_url.rstrip('/')}/" if site_url else None
    public_metadata = ""
    if canonical_url:
        public_metadata = f"""
  <link rel="canonical" href="{esc(canonical_url)}">
  <meta property="og:url" content="{esc(canonical_url)}">"""
    source_link = ""
    if source_url:
        source_link = f' <a href="{esc(source_url)}">Source and reuse terms</a>.'
    if len(history) > 1:
        payload = [
            {
                "date": snapshot.date.isoformat(),
                "summary_html": render_summary_card(
                    snapshot.items, axis_range=axis_range, roster=roster, as_of=snapshot.date
                ),
                "cards_html": render_timeline_cards(snapshot.items, as_of=snapshot.date),
            }
            for snapshot in history
        ]
        encoded_payload = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
        last_index = len(payload) - 1
        history_controls = f"""    <section class="history-controls" aria-label="Historical forecast controls">
      <div class="history-control-row">
        <strong id="history-date">{esc(pretty_date(history[-1].date))}</strong>
        <span class="history-actions">
          <button type="button" id="history-prev" aria-label="Previous day">← Previous</button>
          <button type="button" id="history-play" aria-label="Play forecast history">Play</button>
          <button type="button" id="history-next" aria-label="Next day">Next →</button>
          <button type="button" id="history-latest" aria-label="Jump to latest forecast date">Latest</button>
        </span>
      </div>
      <label for="history-slider">Forecast date</label>
      <input id="history-slider" type="range" min="0" max="{last_index}" value="{last_index}" step="1" aria-valuetext="{esc(pretty_date(history[-1].date))}">
    </section>"""
        history_data = f'<script type="application/json" id="history-data">{encoded_payload}</script>'
        generated_source = "daily Polymarket price history and live quotes"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{esc(SITE_TITLE)}</title>
  <meta name="description" content="{esc(SITE_DESCRIPTION)}">
  <meta property="og:type" content="website">
  <meta property="og:title" content="{esc(SITE_TITLE)}">
  <meta property="og:description" content="{esc(SITE_DESCRIPTION)}">
  <meta name="twitter:card" content="summary">
  <meta name="twitter:title" content="{esc(SITE_TITLE)}">
  <meta name="twitter:description" content="{esc(SITE_DESCRIPTION)}">{public_metadata}
  <style>
    :root {{ color-scheme: light dark; --bg: #f4f5f7; --card: #fff; --text: #17202a; --muted: #667085; --line: #cfd5df; --link: #175cd3; --openai: #10a37f; --anthropic: #c26b3a; --google: #4285f4; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: var(--bg); color: var(--text); font: 16px/1.55 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    main {{ width: min(920px, calc(100% - 32px)); margin: 0 auto; padding: 56px 0 72px; }}
    .page-header {{ margin: 0 0 38px 46px; }}
    h1 {{ margin: 0 0 8px; font-size: clamp(2rem, 6vw, 3.4rem); line-height: 1.05; letter-spacing: -0.04em; }}
    .generated, .note, .detail, .kind {{ color: var(--muted); }}
    .generated {{ margin: 0; }}
    .history-controls {{ margin: -18px 0 30px 46px; padding: 16px 18px; border: 1px solid var(--line); border-radius: 12px; background: var(--card); }}
    .history-control-row {{ display: flex; align-items: center; justify-content: space-between; gap: 14px; margin-bottom: 10px; }}
    .history-actions {{ display: flex; flex-wrap: wrap; gap: 7px; }}
    .history-controls button {{ padding: 6px 10px; border: 1px solid var(--line); border-radius: 7px; background: var(--bg); color: var(--text); font: inherit; font-size: .82rem; cursor: pointer; }}
    .history-controls button:hover:not(:disabled) {{ border-color: var(--muted); }}
    .history-controls button:disabled {{ cursor: default; opacity: .45; }}
    .history-controls label {{ display: block; margin-bottom: 3px; color: var(--muted); font-size: .76rem; }}
    .history-controls input[type="range"] {{ width: 100%; accent-color: var(--link); }}
    .summary-card {{ margin: 0 0 38px 46px; padding: 22px 24px; border: 1px solid var(--line); border-radius: 14px; background: var(--card); box-shadow: 0 5px 20px rgb(16 24 40 / 7%); }}
    .summary-card h2 {{ margin: 0; }}
    .summary-heading {{ display: flex; align-items: center; justify-content: space-between; gap: 14px; }}
    .overview-filter {{ display: inline-flex; flex-wrap: wrap; gap: 5px; }}
    .overview-filter button {{ padding: 5px 9px; border: 1px solid var(--line); border-radius: 7px; background: var(--bg); color: var(--text); font: inherit; font-size: .76rem; cursor: pointer; }}
    .overview-filter button[aria-pressed="true"] {{ border-color: var(--link); background: var(--link); color: var(--card); }}
    #summary-container[data-overview-filter="active"] .summary-plot-row.unavailable,
    #summary-container[data-overview-filter="active"] .summary-table-row.unavailable {{ display: none; }}
    .summary-plot-title {{ margin: 12px 0 10px; color: var(--muted); font-size: .8rem; text-transform: uppercase; letter-spacing: .06em; }}
    .summary-plot-note {{ margin: -4px 0 10px; color: var(--muted); font-size: .76rem; }}
    .summary-plot {{ position: relative; }}
    .summary-plot-row, .summary-axis, .summary-as-of-overlay {{ display: grid; grid-template-columns: minmax(130px, 210px) 1fr; align-items: center; gap: 14px; min-width: 0; }}
    .summary-plot-row {{ min-height: 28px; }}
    .summary-plot-row.unavailable, .summary-table-row.unavailable {{ opacity: .32; }}
    .summary-plot-label {{ overflow: hidden; color: var(--text); font-size: .82rem; text-overflow: ellipsis; white-space: nowrap; }}
    .summary-track {{ position: relative; display: block; height: 24px; }}
    .summary-track::before {{ content: ""; position: absolute; top: 50%; right: 0; left: 0; height: 1px; background: var(--line); }}
    .summary-month-grid {{ position: absolute; top: 0; bottom: 0; width: 1px; background: var(--line); opacity: .85; }}
    .summary-as-of-overlay {{ position: absolute; z-index: 3; inset: 0 0 27px; pointer-events: none; }}
    .summary-as-of-area {{ position: relative; align-self: stretch; }}
    .summary-as-of {{ position: absolute; top: 0; bottom: 0; width: 2px; transform: translateX(-50%); background: var(--text); opacity: .55; pointer-events: auto; }}
    .summary-iqr {{ position: absolute; z-index: 1; top: 50%; height: 2px; transform: translateY(-50%); border-radius: 1px; background: var(--series, var(--muted)); }}
    .summary-iqr.open-left::before, .summary-iqr.open-right::after {{ content: ""; position: absolute; top: 50%; width: 0; height: 0; transform: translateY(-50%); border-top: 5px solid transparent; border-bottom: 5px solid transparent; }}
    .summary-iqr.open-left::before {{ left: -5px; border-right: 6px solid var(--series, var(--muted)); }}
    .summary-iqr.open-right::after {{ right: -5px; border-left: 6px solid var(--series, var(--muted)); }}
    .summary-dot {{ position: absolute; z-index: 2; top: 50%; width: 11px; height: 11px; transform: translate(-50%, -50%); border: 2px solid var(--series, var(--muted)); background: var(--series, var(--muted)); }}
    .summary-dot.median {{ border-radius: 50%; }}
    .summary-dot.horizon {{ transform: translate(-50%, -50%) rotate(45deg); background: var(--card); }}
    .summary-dot.deadline {{ background: var(--card); }}
    .summary-axis {{ margin-top: 1px; }}
    .summary-axis-track {{ position: relative; display: block; height: 26px; border-top: 1px solid var(--line); }}
    .summary-tick {{ position: absolute; top: 5px; color: var(--muted); font-size: .72rem; white-space: nowrap; }}
    .summary-table-wrap {{ margin-top: 18px; overflow-x: auto; }}
    .summary-table {{ width: 100%; border-collapse: collapse; font-size: .86rem; }}
    .summary-table th, .summary-table td {{ padding: 8px 10px; border-top: 1px solid var(--line); text-align: left; vertical-align: top; }}
    .summary-table th {{ color: var(--muted); font-size: .75rem; text-transform: uppercase; letter-spacing: .05em; }}
    .summary-table th:first-child, .summary-table td:first-child {{ padding-left: 0; }}
    .summary-table th:last-child, .summary-table td:last-child {{ padding-right: 0; }}
    .timeline {{ position: relative; margin: 0; padding: 0; list-style: none; }}
    .timeline::before {{ content: ""; position: absolute; top: 12px; bottom: 12px; left: 16px; width: 2px; background: var(--line); }}
    .timeline-item {{ position: relative; padding: 0 0 28px 46px; scroll-margin-top: 16px; }}
    .timeline-item:target article {{ outline: 2px solid var(--link); outline-offset: 3px; }}
    .marker {{ position: absolute; z-index: 1; top: 24px; left: 10px; width: 14px; height: 14px; border: 3px solid var(--card); border-radius: 50%; background: var(--muted); box-shadow: 0 0 0 2px var(--line); }}
    .openai {{ --series: var(--openai); }} .anthropic {{ --series: var(--anthropic); }} .google {{ --series: var(--google); }}
    .openai .marker {{ background: var(--openai); }} .anthropic .marker {{ background: var(--anthropic); }} .google .marker {{ background: var(--google); }}
    article {{ padding: 22px 24px; border: 1px solid var(--line); border-radius: 14px; background: var(--card); box-shadow: 0 5px 20px rgb(16 24 40 / 7%); }}
    .card-header {{ display: flex; align-items: start; justify-content: space-between; gap: 16px; }}
    time {{ display: block; font-size: 1.05rem; font-weight: 750; }}
    .provider {{ display: inline-block; margin-top: 5px; font-size: .76rem; font-weight: 750; letter-spacing: .08em; text-transform: uppercase; color: var(--muted); }}
    .volume {{ white-space: nowrap; font-size: .82rem; color: var(--muted); }}
    h2 {{ margin: 18px 0 7px; font-size: 1.35rem; line-height: 1.25; }}
    a {{ color: var(--link); text-decoration-thickness: .08em; text-underline-offset: .16em; }}
    .estimate {{ margin: 0; font-weight: 700; }}
    .detail {{ margin: 8px 0 0; }}
    .sources {{ margin: 8px 0 0; font-size: .86rem; color: var(--muted); }}
    .kind {{ margin: 13px 0 0; font-size: .8rem; }}
    .prob-chart {{ display: block; width: 100%; height: auto; margin-top: 18px; overflow: visible; }}
    .prob-chart .grid-line {{ stroke: var(--line); stroke-width: 1; }}
    .prob-chart .month-grid {{ stroke: var(--line); stroke-width: 1; opacity: .85; }}
    .prob-chart .threshold {{ stroke-dasharray: 4 4; }}
    .prob-chart .cdf-line {{ fill: none; stroke: var(--series); stroke-width: 2.5; stroke-linejoin: round; stroke-linecap: round; }}
    .prob-chart .cdf-point {{ fill: var(--card); stroke: var(--series); stroke-width: 2; }}
    .prob-chart .pdf-bar {{ fill: var(--series); opacity: .48; }}
    [data-chart-tooltip] {{ cursor: default; }}
    .prob-chart .median-line {{ stroke: var(--text); stroke-width: 1; stroke-dasharray: 3 3; opacity: .7; }}
    .prob-chart .as-of-line {{ stroke: var(--text); stroke-width: 1.5; opacity: .55; pointer-events: stroke; }}
    .prob-chart .as-of-label {{ fill: var(--text); font-size: 11px; font-weight: 500; opacity: .72; }}
    .prob-chart text {{ fill: var(--muted); font-family: inherit; }}
    .prob-chart .plot-label {{ font-size: 12px; font-weight: 700; }}
    .prob-chart .axis-label, .prob-chart .median-label {{ font-size: 11px; }}
    .prob-chart .axis-tick {{ stroke: var(--line); }}
    .chart-tooltip {{ position: absolute; z-index: 100; max-width: min(320px, calc(100vw - 24px)); padding: 8px 10px; border-radius: 7px; background: var(--text); color: var(--card); font-size: .8rem; line-height: 1.35; pointer-events: none; box-shadow: 0 4px 14px rgb(16 24 40 / 20%); }}
    .chart-tooltip[hidden] {{ display: none; }}
    .site-footer {{ margin: 16px 0 0 46px; font-size: .88rem; color: var(--muted); }}
    .site-footer p {{ margin: 8px 0 0; }}
    @media (prefers-color-scheme: dark) {{ :root {{ --bg: #101318; --card: #191e26; --text: #eef1f5; --muted: #a7b0bf; --line: #343c49; --link: #82b1ff; }} }}
    @media (max-width: 600px) {{ main {{ width: min(100% - 20px, 920px); padding-top: 32px; }} .page-header, .history-controls, .summary-card, .site-footer {{ margin-left: 34px; }} .history-control-row {{ align-items: start; flex-direction: column; }} .summary-card {{ padding: 18px; }} .summary-heading {{ align-items: start; flex-direction: column; }} .summary-plot-row, .summary-axis, .summary-as-of-overlay {{ grid-template-columns: minmax(88px, 110px) 1fr; gap: 9px; }} .summary-table {{ min-width: 540px; }} .timeline::before {{ left: 10px; }} .timeline-item {{ padding-left: 34px; }} .marker {{ left: 4px; }} article {{ padding: 18px; }} .card-header {{ display: block; }} .volume {{ display: block; margin-top: 8px; }} }}
  </style>
</head>
<body>
  <main>
    <header class="page-header">
      <h1>{esc(SITE_TITLE)}</h1>
      <p class="generated">Generated {esc(generated_text)} from {generated_source}.</p>
    </header>
{history_controls}
    <div id="summary-container" data-overview-filter="active">{summary_card}</div>
    <ol class="timeline">
{cards_html}
    </ol>
    <footer class="site-footer">
      <p>Median estimates use cumulative market prices and linear interpolation. Historical snapshots use daily traded prices; the latest snapshot uses current quotes and bid/ask bounds. Exclusive date markets are normalized. Prices are forecasts, not facts, and can be noisy or internally inconsistent.</p>
      <p>Independent project; not affiliated with Polymarket, OpenAI, Anthropic, or Google. This is not an official product roadmap or financial advice.{source_link}</p>
      <p>Project code and original presentation © 2026 Matthew Lloyd. All rights reserved; no license is granted for reuse. Polymarket market data remains third-party content.</p>
    </footer>
  </main>
  <div id="chart-tooltip" class="chart-tooltip" role="tooltip" hidden></div>
  {history_data}
  <script>
    (() => {{
      const tooltip = document.getElementById("chart-tooltip");

      function place(target) {{
        const rect = target.getBoundingClientRect();
        const tooltipRect = tooltip.getBoundingClientRect();
        const pageLeft = window.scrollX;
        const pageTop = window.scrollY;
        const minLeft = pageLeft + 8;
        const maxLeft = pageLeft + window.innerWidth - tooltipRect.width - 8;
        let left = pageLeft + rect.left + rect.width / 2 - tooltipRect.width / 2;
        left = Math.max(minLeft, Math.min(maxLeft, left));
        let top = pageTop + rect.top - tooltipRect.height - 9;
        if (top < pageTop + 8) top = pageTop + rect.bottom + 9;
        tooltip.style.left = `${{left}}px`;
        tooltip.style.top = `${{top}}px`;
      }}

      function show(target) {{
        tooltip.textContent = target.dataset.chartTooltip;
        tooltip.hidden = false;
        place(target);
      }}

      function hide() {{ tooltip.hidden = true; }}

      const summary = document.getElementById("summary-container");
      let overviewFilter = "active";

      function applyOverviewFilter(value) {{
        overviewFilter = value === "active" ? "active" : "all";
        summary.dataset.overviewFilter = overviewFilter;
        summary.querySelectorAll("[data-overview-filter]").forEach((button) => {{
          button.setAttribute("aria-pressed", String(button.dataset.overviewFilter === overviewFilter));
        }});
      }}

      document.addEventListener("click", (event) => {{
        const button = event.target.closest?.("[data-overview-filter]");
        if (button) applyOverviewFilter(button.dataset.overviewFilter);
      }});

      document.addEventListener("mouseover", (event) => {{
        const target = event.target.closest?.("[data-chart-tooltip]");
        if (target) show(target);
      }});
      document.addEventListener("mouseout", (event) => {{
        const target = event.target.closest?.("[data-chart-tooltip]");
        if (target && !target.contains(event.relatedTarget)) hide();
      }});
      window.addEventListener("scroll", hide, {{ passive: true }});
      window.addEventListener("resize", hide);

      const historyNode = document.getElementById("history-data");
      if (historyNode) {{
        const snapshots = JSON.parse(historyNode.textContent);
        const slider = document.getElementById("history-slider");
        const previous = document.getElementById("history-prev");
        const next = document.getElementById("history-next");
        const latest = document.getElementById("history-latest");
        const play = document.getElementById("history-play");
        const dateLabel = document.getElementById("history-date");
        const timeline = document.querySelector(".timeline");
        let timer = null;

        function displayDate(value) {{
          return new Date(`${{value}}T00:00:00`).toLocaleDateString(undefined, {{
            year: "numeric", month: "short", day: "numeric"
          }});
        }}

        function renderHistory(index) {{
          const snapshot = snapshots[index];
          summary.innerHTML = snapshot.summary_html;
          timeline.innerHTML = snapshot.cards_html;
          slider.value = index;
          const label = displayDate(snapshot.date);
          dateLabel.textContent = label;
          slider.setAttribute("aria-valuetext", label);
          previous.disabled = index === 0;
          next.disabled = index === snapshots.length - 1;
          latest.disabled = index === snapshots.length - 1;
          applyOverviewFilter(overviewFilter);
          hide();
        }}

        function stopPlayback() {{
          if (timer !== null) window.clearInterval(timer);
          timer = null;
          play.textContent = "Play";
          play.setAttribute("aria-label", "Play forecast history");
        }}

        slider.addEventListener("input", () => {{ stopPlayback(); renderHistory(Number(slider.value)); }});
        previous.addEventListener("click", () => {{ stopPlayback(); renderHistory(Math.max(0, Number(slider.value) - 1)); }});
        next.addEventListener("click", () => {{ stopPlayback(); renderHistory(Math.min(snapshots.length - 1, Number(slider.value) + 1)); }});
        latest.addEventListener("click", () => {{ stopPlayback(); renderHistory(snapshots.length - 1); }});
        play.addEventListener("click", () => {{
          if (timer !== null) {{ stopPlayback(); return; }}
          if (Number(slider.value) === snapshots.length - 1) renderHistory(0);
          play.textContent = "Pause";
          play.setAttribute("aria-label", "Pause forecast history");
          timer = window.setInterval(() => {{
            const index = Number(slider.value);
            if (index >= snapshots.length - 1) {{ stopPlayback(); return; }}
            renderHistory(index + 1);
          }}, 470);
        }});
        renderHistory(snapshots.length - 1);
      }}
      applyOverviewFilter(overviewFilter);
    }})();
  </script>
</body>
</html>"""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", action="append", default=[], help="Add a discovery search query (repeatable).")
    parser.add_argument("--slug", action="append", default=[], help="Include an event slug or Polymarket event URL.")
    parser.add_argument("--min-volume", type=float, default=0.0, help="Hide events below this total USD volume.")
    parser.add_argument(
        "--state-file",
        type=Path,
        default=default_discovery_state_path(),
        help="Persistent discovery state path (default: ~/.polymarket-model-timeline/discovery-state.json).",
    )
    parser.add_argument("--no-state", action="store_true", help="Do not read or update persistent discovery state.")
    parser.add_argument(
        "--history-days",
        type=int,
        default=30,
        help="Daily snapshots embedded in HTML output (default: 30; use 0 to disable).",
    )
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--json", action="store_true", help="Emit machine-readable JSON instead of Markdown.")
    output.add_argument("--html", action="store_true", help="Emit a standalone HTML document instead of Markdown.")
    output.add_argument(
        "--discovery-report",
        action="store_true",
        help="Emit an audit of accepted, rejected, unsupported, new, and changed candidates.",
    )
    parser.add_argument("--site-url", help="Public site URL used for canonical and sharing metadata.")
    parser.add_argument("--source-url", help="Public source repository URL linked from the HTML footer.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.history_days < 0:
        print("error: --history-days must be zero or greater", file=sys.stderr)
        return 2
    queries = tuple(dict.fromkeys((*DEFAULT_SEARCH_QUERIES, *args.query)))
    state_path = None if args.no_state else args.state_file.expanduser()
    try:
        state = load_discovery_state(state_path)
        discovery = discover_event_candidates(queries, args.slug, state)
        events = list(discovery.events)
        items = build_timeline(events, args.min_volume)
        history = (
            build_daily_history(list(discovery.history_events), items, args.history_days, args.min_volume)
            if args.html and args.history_days > 0
            else ()
        )
        save_discovery_state(state_path, discovery.state)
    except (ApiError, StateError, OSError, urllib.error.URLError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    generated_at = dt.datetime.now().astimezone()
    if args.discovery_report:
        print(render_discovery_report(discovery, generated_at))
    elif args.json:
        print(
            json.dumps(
                {"generated_at": generated_at.isoformat(), "items": [item.as_dict() for item in items]},
                indent=2,
            )
        )
    elif args.html:
        print(render_html(items, generated_at, history, site_url=args.site_url, source_url=args.source_url))
    else:
        print(render_markdown(items, generated_at))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
