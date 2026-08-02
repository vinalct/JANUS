"""Concurrent pagination produces exactly what sequential pagination produces."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from conftest import (
    EMPTY_PAGE_SCRIPT,
    PageScript,
    ScriptedPageTransport,
    build_concurrent_plan,
    build_concurrent_strategy,
    build_storage_layout,
)

from janus.checkpoints import ExtractionProgressStore
from janus.models import ExecutionPlan, ExtractionResult
from janus.strategies.api.core import CONCURRENCY_ONLY_METADATA_KEYS
from janus.writers import SIDECAR_SUFFIX, RawArtifactWriter

#: Metadata whose value is scoped to *this* run or *this* concurrency setting, so it can never
#: participate in an equivalence assertion: ``raw_path_prefix`` carries the run id (each leg gets
#: its own by construction) and ``pagination_concurrency`` reports the knob under test.
#: Unlike CONCURRENCY_ONLY_METADATA_KEYS these keys are present in *both* legs — only the values
#: differ — so they are excluded from the value comparison, not from the key comparison.
RUN_SCOPED_METADATA_KEYS = frozenset({"raw_path_prefix", "pagination_concurrency"})

#: Concurrency levels every fixture is proven at. 2 is the smallest speculative window; 5 is
#: wider than any fixture is long, so the whole stream is in flight at once.
CONCURRENCY_LEVELS = (2, 5)

PAGE_SIZE = 2


# ---------------------------------------------------------------------------
# Fingerprint — everything a downstream consumer can observe about an extraction
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ExtractionFingerprint:
    """Everything a downstream consumer can observe about an extraction."""

    artifact_paths: tuple[str, ...]
    artifact_checksums: tuple[str, ...]
    artifact_bytes: tuple[bytes, ...]
    records_extracted: int
    checkpoint_value: str | None
    stable_metadata: tuple[tuple[str, str], ...]

    def without_terminal_page(self) -> ExtractionFingerprint:
        """Drop the trailing empty page a sequential leg has to store to learn it is done.

        Only meaningful for the past-end fixtures (see the module docstring): the sequential leg
        materializes end-of-stream as one extra empty artifact and one extra committed request,
        where the concurrent leg reads a ``404``/``416`` that is not a page at all. Everything
        else — records, checkpoint, and every other metadata key — is untouched, because an empty
        page contributes nothing to any of them.
        """
        metadata = dict(self.stable_metadata)
        for key in ("request_count", "attempt_count"):
            metadata[key] = str(int(metadata[key]) - 1)
        return replace(
            self,
            artifact_paths=self.artifact_paths[:-1],
            artifact_checksums=self.artifact_checksums[:-1],
            artifact_bytes=self.artifact_bytes[:-1],
            stable_metadata=tuple(sorted(metadata.items())),
        )


def fingerprint(result: ExtractionResult, raw_root: Path) -> ExtractionFingerprint:
    """Fingerprint one extraction, with artifact paths relative to ``raw_root``.

    ``raw_root`` is the run-scoped raw directory, because the raw layout is run-scoped
    (``.../ingestion_date=…/run_id=…``) and the two legs differ in ``run_id`` by construction.
    """
    paths: list[str] = []
    checksums: list[str] = []
    payloads: list[bytes] = []
    for artifact in result.artifacts:
        path = Path(artifact.path)
        paths.append(path.relative_to(raw_root).as_posix())
        checksums.append(artifact.checksum)
        payloads.append(path.read_bytes())

    metadata = result.metadata_as_dict()
    stable_metadata = tuple(
        sorted(
            (key, value)
            for key, value in metadata.items()
            if key not in CONCURRENCY_ONLY_METADATA_KEYS and key not in RUN_SCOPED_METADATA_KEYS
        )
    )
    return ExtractionFingerprint(
        artifact_paths=tuple(paths),
        artifact_checksums=tuple(checksums),
        artifact_bytes=tuple(payloads),
        records_extracted=result.records_extracted or 0,
        checkpoint_value=result.checkpoint_value,
        stable_metadata=stable_metadata,
    )


# ---------------------------------------------------------------------------
# Scripted APIs and the two-leg harness
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TransportScript:
    """A complete scripted API: what each pagination key answers, and what unscripted keys do."""

    scripts: Mapping[int, PageScript] = field(default_factory=dict)
    default_script: PageScript = EMPTY_PAGE_SCRIPT
    key_param: str = "page"
    scope_param: str | None = None
    scoped_scripts: Mapping[str, Mapping[int, PageScript]] | None = None

    def build(self) -> ScriptedPageTransport:
        return ScriptedPageTransport(
            self.scripts,
            key_param=self.key_param,
            default_script=self.default_script,
            scope_param=self.scope_param,
            scoped_scripts=self.scoped_scripts,
        )

    def with_descending_latency(self, step: float = 0.05) -> TransportScript:
        """Make later keys answer *faster*, forcing completion order to invert commit order.

        This is how FR-4 is exercised without a wall-clock assertion: the assertion stays on the
        committed artifact order, which must not move when the last page answers first.
        """
        ordered = sorted(self.scripts.items())
        return replace(
            self,
            scripts={
                key: replace(script, latency_seconds=step * (len(ordered) - position))
                for position, (key, script) in enumerate(ordered)
            },
            default_script=replace(self.default_script, latency_seconds=step / 5),
        )


@dataclass(frozen=True, slots=True)
class ExtractionLeg:
    """One leg of an equivalence pair: its fingerprint plus what the run itself observed."""

    fingerprint: ExtractionFingerprint
    metadata: dict[str, str]
    result: ExtractionResult
    transport: ScriptedPageTransport
    run_dir: Path


def run_fixture(
    tmp_path: Path,
    script: TransportScript,
    *,
    concurrency: int,
    label: str,
    prepare: Callable[[ExecutionPlan, Path], None] | None = None,
    attributes: Mapping[str, str] | None = None,
    **plan_kwargs: Any,
) -> ExtractionLeg:
    """Run one extraction against a fresh ``ScriptedPageTransport`` and fingerprint it.

    Each leg gets its own project root, its own ``run_id`` and its own raw root, so the two runs
    of a pair cannot see each other's artifacts. ``prepare`` runs after the plan is built and
    before extraction, for fixtures that need pre-existing on-disk state (resume).
    """
    root = tmp_path / label
    root.mkdir(parents=True, exist_ok=True)
    plan = build_concurrent_plan(
        root,
        source_id=f"equivalence_{label}",
        page_size=PAGE_SIZE,
        concurrency=concurrency,
        attributes=attributes,
        **plan_kwargs,
    )
    if prepare is not None:
        prepare(plan, root)

    transport = script.build()
    strategy = build_concurrent_strategy(root, transport)
    result = strategy.extract(plan)

    run_dir = _run_dir(plan, root, result)
    return ExtractionLeg(
        fingerprint=fingerprint(result, run_dir),
        metadata=result.metadata_as_dict(),
        result=result,
        transport=transport,
        run_dir=run_dir,
    )


def _run_dir(plan: ExecutionPlan, root: Path, result: ExtractionResult) -> Path:
    """Return the run-scoped raw directory this extraction wrote into."""
    raw_root = build_storage_layout(root).resolve_output(plan, "raw").resolved_path
    raw_path_prefix = result.metadata_as_dict().get("raw_path_prefix")
    if raw_path_prefix:
        return raw_root / raw_path_prefix
    return raw_root


def _full_pages(
    count: int,
    *,
    extra_payload: Mapping[str, Any] | None = None,
    first_key: int = 1,
    step: int = 1,
) -> dict[int, PageScript]:
    """Script ``count`` full pages of ``PAGE_SIZE`` records, keyed from ``first_key``.

    ``first_key``/``step`` cover the offset paginator, whose keys are 0, 2, 4, … rather than
    1, 2, 3 — the record ids stay the same either way, so a page-number and an offset fixture
    carry byte-identical payloads.
    """
    scripts: dict[int, PageScript] = {}
    for page in range(count):
        scripts[first_key + page * step] = PageScript(
            records=tuple(
                {"id": str(page * PAGE_SIZE + position + 1)} for position in range(PAGE_SIZE)
            ),
            extra_payload=dict(extra_payload or {}),
        )
    return scripts


# ---------------------------------------------------------------------------
# The fixture matrix
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EquivalenceFixture:
    """One scripted API, plus how its sequential leg has to differ (if at all)."""

    concurrent: TransportScript
    sequential: TransportScript | None = None
    plan_kwargs: Mapping[str, Any] = field(default_factory=dict)
    #: True when the sequential leg stores the terminal empty ``200`` the concurrent leg replaces
    #: with a past-end status. See ``ExtractionFingerprint.without_terminal_page``.
    sequential_commits_terminal_page: bool = False

    @property
    def sequential_script(self) -> TransportScript:
        return self.sequential if self.sequential is not None else self.concurrent


OFFSET_PLAN_KWARGS: Mapping[str, Any] = {
    "variant": "offset_api",
    "pagination_type": "offset",
}

FIXTURES: Mapping[str, EquivalenceFixture] = {
    # The currently-safe precondition: a well-behaved API answering an empty 200 past the end.
    "exact_multiple_empty_200": EquivalenceFixture(
        concurrent=TransportScript(scripts=_full_pages(3)),
    ),
    "short_last_page": EquivalenceFixture(
        concurrent=TransportScript(
            scripts={**_full_pages(2), 3: PageScript(records=({"id": "5"},))},
        ),
    ),
    # The new end-of-stream path. Sequential cannot read a 404 as an ending, so its leg
    # runs the empty-200 tail and the comparison drops that one extra page.
    "past_end_404": EquivalenceFixture(
        concurrent=TransportScript(
            scripts=_full_pages(3),
            default_script=PageScript(status_code=404),
        ),
        sequential=TransportScript(scripts=_full_pages(3)),
        sequential_commits_terminal_page=True,
    ),
    # The second concurrency-capable paginator, on the second default past-end status.
    "past_end_416_offset": EquivalenceFixture(
        concurrent=TransportScript(
            scripts=_full_pages(3, first_key=0, step=PAGE_SIZE),
            key_param="offset",
            default_script=PageScript(status_code=416),
        ),
        sequential=TransportScript(
            scripts=_full_pages(3, first_key=0, step=PAGE_SIZE),
            key_param="offset",
        ),
        plan_kwargs=OFFSET_PLAN_KWARGS,
        sequential_commits_terminal_page=True,
    ),
    # The ceiling path: a reported total caps look-ahead without changing a single artifact.
    "total_count_exposed": EquivalenceFixture(
        concurrent=TransportScript(
            scripts=_full_pages(3, extra_payload={"total": 3 * PAGE_SIZE}),
        ),
    ),
    # The degenerate case: the first index is never speculative, so nothing is ever inferred.
    "single_page": EquivalenceFixture(
        concurrent=TransportScript(scripts={1: PageScript(records=({"id": "1"},))}),
    ),
    # Zero-record runs stay equal — an empty page is still a page, and is still stored.
    "empty_first_page": EquivalenceFixture(
        concurrent=TransportScript(scripts={1: PageScript(records=())}),
    ),
}


def _run_pair(
    tmp_path: Path,
    fixture: EquivalenceFixture,
    *,
    name: str,
    concurrency: int,
    concurrent_script: TransportScript | None = None,
) -> tuple[ExtractionLeg, ExtractionLeg]:
    """Run both legs of ``fixture`` and return them as ``(sequential, concurrent)``."""
    sequential = run_fixture(
        tmp_path,
        fixture.sequential_script,
        concurrency=1,
        label=f"{name}_sequential",
        **fixture.plan_kwargs,
    )
    concurrent = run_fixture(
        tmp_path,
        concurrent_script if concurrent_script is not None else fixture.concurrent,
        concurrency=concurrency,
        label=f"{name}_concurrent_{concurrency}",
        **fixture.plan_kwargs,
    )
    return sequential, concurrent


def _expected(sequential: ExtractionLeg, fixture: EquivalenceFixture) -> ExtractionFingerprint:
    if fixture.sequential_commits_terminal_page:
        return sequential.fingerprint.without_terminal_page()
    return sequential.fingerprint


# ---------------------------------------------------------------------------
# The property
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("concurrency", CONCURRENCY_LEVELS)
@pytest.mark.parametrize("fixture_name", sorted(FIXTURES))
def test_concurrent_matches_sequential(tmp_path, fixture_name, concurrency):
    """Same script, two concurrency settings, byte-identical observable output."""
    fixture = FIXTURES[fixture_name]
    sequential, concurrent = _run_pair(
        tmp_path,
        fixture,
        name=fixture_name,
        concurrency=concurrency,
    )

    assert concurrent.fingerprint.artifact_paths, "fixture produced no artifacts to compare"
    assert concurrent.fingerprint == _expected(sequential, fixture)


@pytest.mark.parametrize("fixture_name", sorted(FIXTURES))
def test_concurrent_run_is_actually_concurrent(tmp_path, fixture_name):
    """Guard against an equivalence suite that passes because concurrency stopped working.

    The cheapest way to make this file green is to break concurrency, so every fixture is also
    run with inverted latencies at ``concurrency=5``: requests must genuinely overlap, and the
    committed order must survive completion order arriving backwards.
    """
    fixture = FIXTURES[fixture_name]
    sequential, concurrent = _run_pair(
        tmp_path,
        fixture,
        name=f"{fixture_name}_latency",
        concurrency=5,
        concurrent_script=fixture.concurrent.with_descending_latency(),
    )

    assert concurrent.transport.max_active_requests >= 2
    assert concurrent.fingerprint == _expected(sequential, fixture)


def test_concurrency_only_metadata_keys_are_the_only_difference(tmp_path):
    """The metadata allowlist is exhaustive: nothing else about the two runs differs."""
    fixture = FIXTURES["exact_multiple_empty_200"]
    sequential, concurrent = _run_pair(
        tmp_path,
        fixture,
        name="metadata_diff",
        concurrency=5,
    )

    keys_in_one_leg_only = set(concurrent.metadata) ^ set(sequential.metadata)
    assert keys_in_one_leg_only == CONCURRENCY_ONLY_METADATA_KEYS & set(concurrent.metadata)
    assert keys_in_one_leg_only, "the concurrent leg reported no speculation metadata at all"

    shared_keys = set(concurrent.metadata) & set(sequential.metadata)
    differing_values = {
        key for key in shared_keys if concurrent.metadata[key] != sequential.metadata[key]
    }
    assert differing_values == RUN_SCOPED_METADATA_KEYS


# ---------------------------------------------------------------------------
# Additional equivalence dimensions
# ---------------------------------------------------------------------------


MULTI_INPUT_PLAN_KWARGS: Mapping[str, Any] = {
    "request_inputs": {
        "type": "date_window",
        "start": date(2025, 1, 1),
        "end": date(2025, 2, 28),
        "step": "month",
    },
    "parameter_bindings": {"mesAno": {"from": "request_input.window_end", "format": "%Y%m"}},
}

MULTI_INPUT_FIXTURE = EquivalenceFixture(
    concurrent=TransportScript(
        scope_param="mesAno",
        scoped_scripts={
            "202501": _full_pages(2),
            "202502": {1: PageScript(records=({"id": "1"},))},
        },
    ),
    plan_kwargs=MULTI_INPUT_PLAN_KWARGS,
)


@pytest.mark.parametrize("concurrency", CONCURRENCY_LEVELS)
def test_multi_request_input_equivalence(tmp_path, concurrency):
    """Per-input artifact naming (``request-input-NNNNNN``) is identical across both legs."""
    sequential, concurrent = _run_pair(
        tmp_path,
        MULTI_INPUT_FIXTURE,
        name="multi_input",
        concurrency=concurrency,
    )

    assert concurrent.fingerprint == sequential.fingerprint
    assert concurrent.fingerprint.artifact_paths == (
        "request-input-000001/page-0001.json",
        "request-input-000001/page-0002.json",
        "request-input-000001/page-0003.json",
        "request-input-000002/page-0001.json",
    )
    assert concurrent.fingerprint.records_extracted == 5


RESUME_FIXTURE = EquivalenceFixture(concurrent=TransportScript(scripts=_full_pages(4)))
RESUMED_FROM_PAGE = 2


def _seed_interrupted_run(plan: ExecutionPlan, root: Path) -> None:
    """Write the artifacts and the progress row a run interrupted after page 2 would leave.

    The progress row carries no ``raw_path_prefix``, which is what pins both the recovered and
    the newly written artifacts to the same ``pages/`` directory — the layout
    ``_rediscover_raw_artifacts`` reads.
    """
    writer = RawArtifactWriter(build_storage_layout(root))
    for page_number in range(1, RESUMED_FROM_PAGE + 1):
        payload = json.loads(RESUME_FIXTURE.concurrent.scripts[page_number].encoded_body())
        writer.write_json(plan, Path("pages") / f"page-{page_number:04d}.json", payload)

    ExtractionProgressStore().save(
        plan,
        page_number=RESUMED_FROM_PAGE,
        request_index=RESUMED_FROM_PAGE,
        artifact_count=RESUMED_FROM_PAGE,
        completed_inputs=[],
        current_input_key="__none__",
        current_input_index=1,
        request_input_count=1,
    )


@pytest.mark.parametrize("concurrency", CONCURRENCY_LEVELS)
def test_resume_equivalence(tmp_path, concurrency):
    """Recovered and newly extracted artifacts are identical on both legs.

    Resuming re-anchors speculation: the resumed input starts at ``request_index=1`` with
    ``page_number=3``, so the policy's ``first_request_index`` — not a hard-coded 1 — decides
    which indexes are guesses.
    """
    legs = {
        leg_concurrency: run_fixture(
            tmp_path,
            RESUME_FIXTURE.concurrent,
            concurrency=leg_concurrency,
            label=f"resume_{leg_concurrency}",
            prepare=_seed_interrupted_run,
            attributes={"resume": "true"},
        )
        for leg_concurrency in (1, concurrency)
    }
    sequential, concurrent = legs[1], legs[concurrency]

    assert concurrent.fingerprint == sequential.fingerprint
    assert concurrent.fingerprint.artifact_paths == (
        "pages/page-0001.json",
        "pages/page-0002.json",
        "pages/page-0003.json",
        "pages/page-0004.json",
        "pages/page-0005.json",
    )
    # Only pages 3-5 were fetched; pages 1-2 were recovered from the interrupted run.
    assert concurrent.fingerprint.records_extracted == 4
    assert concurrent.metadata["request_count"] == "3"
    assert int(concurrent.metadata["speculative_request_count"]) > 0


CHECKPOINT_FIXTURE = EquivalenceFixture(
    concurrent=TransportScript(
        scripts={
            1: PageScript(
                records=(
                    {"id": "1", "updated_at": "2025-01-01"},
                    {"id": "2", "updated_at": "2025-01-02"},
                )
            ),
            2: PageScript(
                records=(
                    {"id": "3", "updated_at": "2025-01-04"},
                    {"id": "4", "updated_at": "2025-01-03"},
                )
            ),
        }
    ),
    plan_kwargs={"checkpoint_field": "updated_at"},
)


@pytest.mark.parametrize("concurrency", CONCURRENCY_LEVELS)
def test_checkpoint_equivalence(tmp_path, concurrency):
    """The final checkpoint is the max over committed records, whatever order they arrived in."""
    sequential, concurrent = _run_pair(
        tmp_path,
        CHECKPOINT_FIXTURE,
        name="checkpoint",
        concurrency=concurrency,
    )

    assert concurrent.fingerprint == sequential.fingerprint
    assert concurrent.fingerprint.checkpoint_value == "2025-01-04"


@pytest.mark.parametrize("concurrency", CONCURRENCY_LEVELS)
def test_raw_sidecar_equivalence(tmp_path, concurrency):
    """Every artifact keeps its ``<path>.sha256`` companion, with the same digest."""
    fixture = FIXTURES["exact_multiple_empty_200"]
    sequential, concurrent = _run_pair(
        tmp_path,
        fixture,
        name="sidecar",
        concurrency=concurrency,
    )

    concurrent_sidecars = _sidecar_digests(concurrent)
    assert concurrent_sidecars == _sidecar_digests(sequential)
    assert tuple(concurrent_sidecars) == concurrent.fingerprint.artifact_paths
    assert tuple(concurrent_sidecars.values()) == concurrent.fingerprint.artifact_checksums


def _sidecar_digests(leg: ExtractionLeg) -> dict[str, str]:
    """Return ``{relative artifact path: sidecar digest}`` for one leg, in committed order."""
    digests: dict[str, str] = {}
    for artifact in leg.result.artifacts:
        path = Path(artifact.path)
        sidecar = path.with_name(path.name + SIDECAR_SUFFIX)
        relative_path = path.relative_to(leg.run_dir).as_posix()
        digests[relative_path] = sidecar.read_text(encoding="utf-8").strip()
    return digests
