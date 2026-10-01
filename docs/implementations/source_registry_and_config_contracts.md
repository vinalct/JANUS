# Source Registry and Config Contracts

The short version is: JANUS now has a real source registry, a typed source configuration model, and a validation layer that stops bad YAML before planner code ever has to touch it.

This is the point where source onboarding starts to look like framework work instead of script work.

## What was added

- `conf/app.yaml` now tells the project where source definition files live.
- `conf/sources/` now exists as the home for source YAML files.
- `src/janus/models/source_config.py` defines the typed contract for a source.
- `src/janus/registry/loader.py` loads the app config, discovers source files, validates them, and returns typed objects.
- `tests/unit/registry/` covers the main happy path and the failure cases that matter for onboarding.

There is also one checked-in example source file at `conf/sources/example/example_source.yaml`.

That file is intentionally not a live integration. It points to `example.invalid` on purpose. The goal here was to lock down the contract shape first, not to sneak source-specific behavior into the registry.

## How discovery works

The registry flow is small by design:

1. Load `conf/app.yaml`.
2. Read the `registry.sources_dir` setting.
3. Discover matching YAML files under that directory recursively.
4. Parse each file into one `SourceConfig` or a grouped `sources:` list of `SourceConfig` entries.
5. Reject the load immediately if any file is malformed.
6. Expose a typed registry API to the next layer.

That means planner code can work with `SourceConfig` objects and does not need to know anything about `yaml.safe_load`, directory walking, or nested config validation.

## The source contract, block by block

A source definition is expected to carry these sections:

- Identity fields such as `source_id`, `name`, `owner`, `domain`, and `enabled`.
- Strategy fields such as `source_type`, `strategy`, and `strategy_variant`.
- An `access` block for connection details, auth, pagination, rate limits, request inputs, and file discovery options.
- An `extraction` block for mode, checkpoint semantics, retry behavior, and the dead-letter budget for item-level self-healing.
- A `schema` block whose only key, `contract`, points at the bronze table's ODCS data contract.
- A `spark` block for input format and write behavior.
- An `outputs` block for `raw`, `bronze`, and `metadata` targets. Each target declares `path` and `format`, and Iceberg bronze targets may optionally declare `namespace` and `table_name`.
- A `quality` block whose `required_fields` / `unique_fields`, when declared, must equal the contract's `required` columns and `primaryKey`. Schema evolution is governed by the contract's `janus.compatibility`.

Two choices are worth calling out:

- `source_type` and `strategy` are kept aligned on purpose. For the current JANUS shape, the strategy family is the source family.
- The registry validates contracts, but it does not try to express source-specific quirks. If a real source does not fit this shape cleanly, that is a framework discussion, not a reason to add ad hoc exceptions here.

## What the validation already catches

The loader fails fast when it sees problems like:

- missing required fields;
- wrong scalar types;
- unsupported strategy families or variants;
- invalid auth combinations;
- incomplete pagination blocks;
- invalid request-input or parameter-binding combinations;
- unsupported file link resolver modes;
- incremental extraction without checkpoint details;
- malformed output targets;
- invalid Iceberg bronze naming fields such as `namespace` or `table_name` outside `outputs.bronze` or outside `format: iceberg`;
- duplicate `source_id` values across files.

Errors come back with dotted field paths such as `access.auth.env_var` or `outputs.raw.path`, so the fix is usually obvious from the message itself.

## What this step does not do yet

This task was deliberately narrow.

It does not:

- choose a strategy implementation;
- build an execution plan;
- call external APIs or download files;
- run Spark work;
- add hooks for special-case sources.

Those pieces belong in later tasks. The point here was to give those later layers a stable contract to build on.

## How to use it from code

The public entry point is the registry loader:

```python
from pathlib import Path

from janus.registry import load_registry


project_root = Path("/path/to/janus")
registry = load_registry(project_root)

enabled_sources = registry.list_sources()
one_source = registry.get_source("federal_open_data_example")
```

From there, downstream code works with typed objects instead of raw dictionaries.

## If you are adding the next source

Start with `conf/sources/example/example_source.yaml`.

Keep these rules in mind:

- stay inside an existing strategy family and variant if the source really fits it;
- keep auth, pagination, request inputs, file discovery, checkpointing, and outputs declarative when possible, including Iceberg bronze naming when a source needs an explicit namespace or table name;
- do not teach the registry about one special source;
- if the contract feels wrong for a real source, stop and raise the design gap instead of papering over it.

That last point matters. A registry that slowly fills up with exceptions turns into a bag of hidden business rules, and then the planner has no clean foundation to stand on.
