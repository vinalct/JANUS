"""Containment: the catalog this order migrates off may only be named where it was sanctioned."""

from __future__ import annotations

from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]

#: The token, matched case-insensitively.
TOKEN = "hadoop"

#: Directory roots walked in full.
SWEPT_DIRECTORIES = ("src", "tests", "conf", "docker")

#: Individual files swept outside those roots. Not all are visible in every environment —
#: the dev container mounts only src/, tests/, conf/ and data/ — so each is filtered by
#: ``is_file()`` rather than assumed present.
SWEPT_FILES = (PROJECT_ROOT / "Makefile",)

#: Directory names never swept — build artefacts, not source.
IGNORED_DIRECTORY_NAMES = frozenset({"__pycache__"})

#: The sites allowed to name the token, each with the reason it is allowed to.
#: Paths are relative to the project root.
ALLOWED_SITES: dict[str, str] = {
    "src/janus/utils/environment.py": (
        "The seam's own vocabulary: HADOOP_CATALOG_TYPE and its place in "
        "SUPPORTED_CATALOG_TYPES. The profile layer must still be able to name the type it "
        "emits options for, because local-hadoop.yaml and the AC-4 differential both select "
        "it. This is the one definition; every other site below reads it from here."
    ),
    "src/janus/utils/catalog_properties.py": (
        "HadoopCatalogUnrepresentableError and the message it raises. pyiceberg implements "
        "no Hadoop catalog — its commits are filesystem renames — and saying so by name is "
        "the engine-neutrality gap that justifies this order. An unnamed error would state "
        "the failure without stating the reason."
    ),
    "conf/environments/local-hadoop.yaml": (
        "The throwaway pre-migration profile itself, kept so the old commit path stays "
        "reproducible for the AC-4 baseline. Its header says it is not the supported local "
        "environment; local.yaml is."
    ),
    "conf/environments/cluster.yaml": (
        "The cluster profile still defaults to the old catalog."
    ),
    "conf/environments/cluster.env.example": (
        "The documented default for the cluster profile above, and the same temporary"
    ),
    "tests/integration/catalog_migration/test_bronze_identity_across_catalogs.py": (
        "The AC-4 differential, and the one test that may build a session on the old "
        "catalog: the old catalog is the baseline half of the comparison, so removing it "
        "would remove the evidence that bronze survived the swap."
    ),
    "tests/unit/utils/test_environment_vendor_neutrality.py": (
        "Names the token only inside a source-text sample the vendor-neutrality detector is "
        "run against, proving the detector fires on it. A detector meta-test input, not a "
        "catalog choice — and it builds no Spark session."
    ),
    "tests/unit/utils/test_catalog_options.py": (
        "Pins the option block `catalog_type: hadoop` still emits (AC-4 for the seam itself) "
        "and that the supported set holds all three types. Asserts over the returned mapping "
        "only; builds no Spark session."
    ),
    "tests/unit/utils/test_catalog_properties.py": (
        "Asserts the named error above, and that _PROPERTY_BUILDERS covers exactly the "
        "supported types minus hadoop — a set difference that cannot be written without "
        "naming it. Builds no Spark session."
    ),
    "tests/unit/toolchain/test_hadoop_containment.py": (
        "This sweep. It must name the token to detect it, and names the session-builder "
        "marker for the same reason; both are matched against its own text, which is why it "
        "is in SESSION_BUILD_EXEMPT as well."
    ),
}

#: The one test module allowed to build a Spark session on the old catalog.
DIFFERENTIAL_MODULE = (
    "tests/integration/catalog_migration/test_bronze_identity_across_catalogs.py"
)

#: This module's own path, resolved rather than spelled, so a rename cannot strand it.
THIS_MODULE = str(Path(__file__).resolve().relative_to(PROJECT_ROOT))

#: Modules allowed to name the token *and* the session-builder marker: the differential,
#: which really does build that session, and this sweep, which names both to detect them.
SESSION_BUILD_EXEMPT = frozenset({DIFFERENTIAL_MODULE, THIS_MODULE})

#: How a Spark session gets built, in any of these files.
SESSION_BUILDER_MARKER = "SparkSession.builder"

#: Sites that are readable wherever the suite runs, container included. The sweep asserts it
#: found each of them, so a wrong root or a broken walk cannot pass by matching nothing.
ALWAYS_VISIBLE_SITES = (
    "src/janus/utils/environment.py",
    "conf/environments/local-hadoop.yaml",
    "tests/unit/toolchain/test_hadoop_containment.py",
)


# ── the detector ─────────────────────────────────────────────────────────────


def token_mentions(text: str) -> list[str]:
    """Every line naming the token, case-insensitively."""

    return [line.strip() for line in text.splitlines() if TOKEN in line.lower()]


# ── the sweep ────────────────────────────────────────────────────────────────


def _read(path: Path) -> str:
    """File text, tolerating anything that is not valid UTF-8 rather than failing on it."""

    return path.read_bytes().decode("utf-8", errors="ignore")


def _swept_paths() -> list[Path]:
    """Every file under the swept roots that exists here."""

    paths = [path for path in SWEPT_FILES if path.is_file()]
    for name in SWEPT_DIRECTORIES:
        root = PROJECT_ROOT / name
        if not root.is_dir():
            continue
        paths.extend(
            path
            for path in sorted(root.rglob("*"))
            if path.is_file()
            and IGNORED_DIRECTORY_NAMES.isdisjoint(path.relative_to(root).parts)
        )
    return paths


def _mentions_by_site() -> dict[str, list[str]]:
    """Swept files naming the token, keyed by their project-relative path."""

    found: dict[str, list[str]] = {}
    for path in _swept_paths():
        mentions = token_mentions(_read(path))
        if mentions:
            found[str(path.relative_to(PROJECT_ROOT))] = mentions
    return found


def test_the_sweep_actually_reads_the_repository():
    """A walk that silently matched nothing would pass every assertion below it."""

    swept = {str(path.relative_to(PROJECT_ROOT)) for path in _swept_paths()}

    assert len(swept) >= 100, f"only {len(swept)} files swept; the walk looks truncated"
    for site in ALWAYS_VISIBLE_SITES:
        assert site in swept, (
            f"the sweep never read {site}, which is readable in every environment — the "
            "roots are wrong and the containment assertions below are vacuous"
        )


def test_every_mention_of_the_token_is_allowlisted():
    """The containment claim: no site names the old catalog without a recorded decision."""

    undeclared = {
        site: mentions
        for site, mentions in _mentions_by_site().items()
        if site not in ALLOWED_SITES
    }

    assert not undeclared, (
        "site(s) naming the catalog this order migrates off, with no entry in "
        "ALLOWED_SITES:\n"
        + "\n".join(
            f"  {site}\n" + "\n".join(f"      {line}" for line in mentions)
            for site, mentions in sorted(undeclared.items())
        )
        + "\nBuild the session through tests/support/spark_sessions.py instead, or add an "
        "entry stating why this site is allowed to name it."
    )


def test_no_allowlisted_site_has_gone_stale():
    """An entry must describe a file that still exists and still names the token.

    Without this the allowlist becomes fiction: an entry granted to a file that has since
    been rewritten would keep documenting a decision nobody is making any more — and the
    two cluster-profile entries would outlive the rewrite that is meant to retire them.
    """

    found = _mentions_by_site()
    visible = {
        site for site in ALLOWED_SITES if (PROJECT_ROOT / site).is_file()
    }
    stale = sorted(site for site in visible if site not in found)

    assert not stale, (
        "ALLOWED_SITES names file(s) that no longer contain the token:\n"
        + "\n".join(f"  {site}" for site in stale)
        + "\nRemove the entry — the exception it records no longer applies."
    )
    for site in ALWAYS_VISIBLE_SITES:
        assert site in visible, f"{site} is missing; the stale check cannot be trusted"


def test_every_allowlist_entry_carries_a_written_reason():
    """An allowance with no reason is a backlog item pretending to be a decision."""

    unreasoned = sorted(site for site, reason in ALLOWED_SITES.items() if not reason.strip())

    assert not unreasoned, (
        "ALLOWED_SITES entr(ies) with no written reason:\n"
        + "\n".join(f"  {site}" for site in unreasoned)
    )


def test_only_the_differential_builds_a_session_on_the_migrated_catalog():
    """The sharper half of the rule, and the one the ~12 ported builders used to break."""

    offenders = sorted(
        site
        for site, _mentions in _mentions_by_site().items()
        if site.startswith("tests/")
        and site not in SESSION_BUILD_EXEMPT
        and SESSION_BUILDER_MARKER in _read(PROJECT_ROOT / site)
    )

    assert not offenders, (
        "test module(s) that both name the old catalog and build a Spark session:\n"
        + "\n".join(f"  {site}" for site in offenders)
        + "\nBuild it through tests/support/spark_sessions.py, which derives its catalog "
        "conf from the profile the way main.py does."
    )


# ── detector meta-tests ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("label", "snippet"),
    [
        ("bare", 'catalog_type: hadoop'),
        ("capitalized", "The Hadoop catalog commits by filesystem rename."),
        ("shouted", "JANUS_ICEBERG_CATALOG_TYPE=HADOOP"),
        ("camel_suffix", "value = hadoopFooBar"),
        ("spark_conf", '.config("spark.sql.catalog.janus.type", "hadoop")'),
        ("java_package", "import org.apache.hadoop.fs.Path;"),
        ("later_line", "name: local\ncatalog_type: hadoop\n"),
    ],
)
def test_the_detector_flags_a_deliberate_mention(label, snippet):
    """Every spelling the sweep claims to cover actually trips it."""

    del label
    assert token_mentions(snippet), f"detector missed a mention in:\n{snippet}"


@pytest.mark.parametrize(
    ("label", "snippet"),
    [
        ("jdbc_profile", "catalog_type: jdbc"),
        ("sqlite_uri", "uri: jdbc:sqlite:data/metadata/iceberg-catalog/catalog.sqlite"),
        ("split_word", "had oop"),
        ("neighbouring_words", "warehouse_dir: data/bronze/iceberg"),
        ("empty", ""),
    ],
)
def test_the_detector_does_not_flag_clean_text(label, snippet):
    """And it does not fire on the configuration that replaced it."""

    del label
    assert token_mentions(snippet) == []
