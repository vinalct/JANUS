"""The one registration point: `janus <verb> …` is resolved here and nowhere else."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from janus.cli import checkpoint, dead_letters, list_sources, maintain, run, run_all, validate
from janus.cli.common import build_parent_parser

PROG = "janus"
VERB_RUN = "run"

_HELP_FLAGS = frozenset({"-h", "--help"})

_DESCRIPTION = """\
Validate JANUS configuration, plan or execute sources, and author data
contracts. Each verb takes its own options; `janus <verb> --help` lists them.

`run` is the default verb when the first argument is an option, so
`janus --environment local --source-id ID --execute` runs one source. Use
`janus run-all [options]` to execute enabled sources once in dependency order."""


@dataclass(frozen=True, slots=True)
class Verb:
    """One subcommand: how to parse it and how to run it.

    A verb declares its options in ``configure``; the dispatcher builds its parser
    (``janus <name>`` over :func:`build_parent_parser`) and ``handler`` receives the parsed
    namespace. A verb whose parser predates the dispatcher leaves ``configure`` as ``None``
    and its handler receives the tokens after the verb unparsed: ``run``, ``run-all`` and
    ``contract`` keep the parsers whose ``--help`` the AC-1 goldens assert.
    """

    name: str
    help: str
    configure: Callable[[argparse.ArgumentParser], None] | None
    handler: Callable[..., int]


def build_parser() -> argparse.ArgumentParser:
    """The top-level parser behind `janus --help`. It lists the verbs and parses none of them."""
    # `prog` is pinned: under `python -m janus.main` argparse would derive it from the launcher.
    parser = argparse.ArgumentParser(
        prog=PROG,
        description=_DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    listing = parser.add_subparsers(title="verbs")
    for verb in verbs():
        listing.add_parser(verb.name, help=verb.help, add_help=False)
    return parser


def verbs() -> tuple[Verb, ...]:
    """The registry. A verb that is not here is not reachable from main()."""
    return (
        Verb(
            VERB_RUN,
            "Validate runtime configuration, plan one source, or execute it. The default "
            "when the first argument is an option.",
            None,
            run.run_command,
        ),
        Verb(
            "run-all",
            "Run enabled sources once in deterministic dependency order.",
            None,
            run_all.main,
        ),
        Verb(
            "contract",
            "Inspect and author versioned data contract files.",
            None,
            _contract,
        ),
        Verb(
            "validate",
            "Load the source registry, resolve its graph, and plan every source, "
            "without running any.",
            validate.configure,
            validate.validate_command,
        ),
        Verb(
            "list",
            "List configured sources with their dispatch, state and dependencies, without "
            "running any.",
            list_sources.configure,
            list_sources.list_command,
        ),
        Verb(
            "dead-letters",
            "Inspect, release or replay the items a source's runs gave up on. Nothing runs "
            "unless replay is given --execute.",
            dead_letters.configure,
            dead_letters.dead_letters_command,
        ),
        Verb(
            "checkpoint",
            "Show where a source's next run starts, or set or clear it with a recorded "
            "reason. Nothing runs.",
            checkpoint.configure,
            checkpoint.checkpoint_command,
        ),
        Verb(
            "maintain",
            "Plan declared retention policies by default, or apply them with recorded evidence.",
            maintain.configure,
            maintain.maintain_command,
        ),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Resolve the verb (implicit `run` when the first token is a flag) and delegate.

    `-h`/`--help` in first position prints the verb list. Any other leading option belongs to
    `run`, so every invocation written before the dispatcher keeps its parser and output.
    """
    resolved_argv = tuple(sys.argv[1:] if argv is None else argv)
    table = {verb.name: verb for verb in verbs()}

    if resolved_argv and resolved_argv[0] in _HELP_FLAGS:
        parser = build_parser()
        parser.print_help()
        parser.exit()
    if not resolved_argv or resolved_argv[0].startswith("-"):
        return _invoke(table[VERB_RUN], resolved_argv)

    verb = table.get(resolved_argv[0])
    if verb is None:
        print(f"{PROG}: unknown command {resolved_argv[0]!r}", file=sys.stderr)
        print(f"choose from: {', '.join(table)}", file=sys.stderr)
        return 2
    return _invoke(verb, resolved_argv[1:])


def _invoke(verb: Verb, argv: Sequence[str]) -> int:
    if verb.configure is None:
        return verb.handler(argv)
    parser = argparse.ArgumentParser(
        prog=f"{PROG} {verb.name}",
        description=verb.help,
        parents=[build_parent_parser()],
        allow_abbrev=False,
    )
    verb.configure(parser)
    return verb.handler(parser.parse_args(argv))


def _contract(argv: Sequence[str]) -> int:
    # Imported on use, as before the dispatcher: no run path loads the drafting stack.
    from janus.cli.contract import main as contract_main

    return contract_main(argv)
