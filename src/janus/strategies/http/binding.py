"""URL / param / checkpoint binding helpers shared by the api and catalog strategies.

Collapses the four binding helpers that api and catalog each carried a private
copy of (``_split_path_and_query_params``, ``_resolve_url``,
``_default_checkpoint_params``, ``_checkpoint_request_value``) into one definition
apiece. 
"""

from __future__ import annotations

import re
from datetime import timedelta
from urllib.parse import urljoin

from janus.checkpoints import CheckpointState
from janus.models import ExecutionPlan, SourceConfig
from janus.strategies.common import _format_datetime, _parse_datetime


def split_path_and_query_params(
    url: str,
    bound_params: dict[str, str],
) -> tuple[dict[str, str], dict[str, str]]:
    """Separate bound params into path params (referenced as {name} in the URL) and query params."""
    placeholders = set(re.findall(r"\{(\w+)\}", url))
    path_params = {k: v for k, v in bound_params.items() if k in placeholders}
    query_params = {k: v for k, v in bound_params.items() if k not in placeholders}
    return path_params, query_params


def resolve_url(source_config: SourceConfig, *, family_label: str) -> str:
    """Resolve the request URL from ``access.url`` or ``access.base_url`` + ``access.path``.

    ``family_label`` (``"API"`` / ``"Catalog"``) names the family in the
    missing-configuration ``ValueError`` so each core keeps its byte-identical message.
    """
    if source_config.access.url:
        return source_config.access.url

    base_url = (source_config.access.base_url or "").strip()
    path = (source_config.access.path or "").strip()
    if not base_url:
        raise ValueError(f"{family_label} source requires access.base_url or access.url")
    if not path:
        return base_url
    return urljoin(base_url.rstrip("/") + "/", path.lstrip("/"))


def default_checkpoint_params(
    plan: ExecutionPlan,
    checkpoint_value: str | None,
) -> dict[str, str]:
    if checkpoint_value is None or plan.checkpoint_field is None:
        return {}
    return {plan.checkpoint_field: checkpoint_value}


def checkpoint_request_value(
    plan: ExecutionPlan,
    checkpoint_state: CheckpointState | None,
) -> str | None:
    if (
        plan.extraction_mode != "incremental"
        or plan.checkpoint_field is None
        or checkpoint_state is None
    ):
        return None

    value = checkpoint_state.checkpoint_value
    lookback_days = plan.source_config.extraction.lookback_days
    if not lookback_days:
        return value

    parsed_datetime = _parse_datetime(value)
    if parsed_datetime is None:
        return value
    return _format_datetime(parsed_datetime - timedelta(days=lookback_days))
