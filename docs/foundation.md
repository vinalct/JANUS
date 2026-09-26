# JANUS
## A Metadata-Driven, Reproducible, Safe, and Extensible Spark Framework for Ingesting Brazilian Federal Open Data

---

## 1. Simple Summary

**JANUS** is a post-graduation project focused on building a **reproducible**, **safe**, **modular**, and **scalable** data engineering framework to ingest **federal Brazilian public data** using **Apache Spark**.

The project will start with a small number of clearly public federal sources (3 to 5), but it will be designed from day one to support growth without turning into a collection of isolated scripts. The central idea is to combine:

- a **strong agnostic foundation** for common ingestion concerns;
- **strategy modules** for distinct source patterns;
- **YAML-driven source definitions** for source onboarding and configuration;
- **limited source-specific code only when a source really behaves differently**.

This is important because public data sources are heterogeneous in practice. Two sources may both be APIs, but still differ in pagination, authentication, payload structure, retry behavior, rate limits, update semantics, and schema evolution. Likewise, a CSV download source behaves very differently from an API source. Therefore, JANUS must not pretend that everything can be solved with only one generic extractor and one YAML schema. Instead, it should provide a clean modular architecture where different extraction strategies are organized explicitly and remain easy to maintain.

The first goal is **not** to ingest all Brazilian public data. The first goal is to prove that JANUS can ingest heterogeneous federal open-data sources in a way that is:

- reproducible in any Spark cluster or local machine;
- safe and respectful of public-data usage constraints;
- easy to extend;
- architecturally clean;
- measurable from a data engineering perspective.

In future stages, JANUS may evolve to support **state** and **municipal** data sources, but the initial scope will be restricted to **federal government public data only**.

---

## 2. Project Vision

JANUS is intended to be both:

1. **a practical data engineering project** for the post-graduation final work; and
2. **a research foundation** that can evolve into a master's and later a PhD project.

The project is motivated by a real problem: public government data is available through different mechanisms, with different formats, different update patterns, different quality levels, and different operational constraints. Some sources are API-based, others are bulk-download based, others are primarily metadata catalogs, and even sources of the same family can behave in different ways.

Because of that, the main technical challenge is not just "download files" or "call APIs". The challenge is to build a **general ingestion framework** that can support heterogeneous public sources with good engineering practices **without forcing all sources into an unrealistic one-size-fits-all model**.

---

## 3. Core Objective

Design and implement a **Spark-based ingestion framework** for **federal Brazilian public data** that is:

- **metadata-driven where appropriate**;
- **modular by extraction strategy**;
- **agnostic at the core, but realistic about source differences**;
- **reproducible on any machine or Spark cluster**;
- **safe by default**;
- **easy to evolve**;
- **cleanly structured to avoid spaghetti code**.

In JANUS, "agnostic design" does **not** mean that all sources will be supported only by YAML and zero code. It means that the project will separate:

- what is **common and reusable** across many sources;
- what belongs to a **strategy family** such as API, file download, or catalog ingestion;
- what is truly **source-specific** and must be isolated in dedicated modules.

---

## 4. Problem Statement

Federal public datasets are exposed through multiple channels such as APIs, downloadable CSV/ZIP files, and open-data catalogs. These sources differ in:

- access protocol;
- authentication needs;
- schema stability;
- update cadence;
- incremental load support;
- data volume;
- rate limits and operational constraints;
- payload format;
- pagination pattern;
- retry behavior;
- response semantics;
- file naming and publication patterns.

Even within a single family such as APIs, different sources may not share the same design. For example:

- one API may use page-based pagination;
- another may use offsets;
- another may use date windows;
- another may expose nested JSON structures with unstable fields;
- another may require request headers or access tokens;
- another may not provide reliable incremental semantics.

Likewise, download-based sources may differ in:

- whether the file is static or versioned;
- whether it is CSV, ZIP, XLSX, or mixed packages;
- whether the schema is stable across releases;
- whether the source exposes one file per period or replaces the same file over time.

When teams try to ingest these sources without a framework, the result is usually:

- source-specific scripts spread across folders;
- duplicated extraction logic;
- hardcoded endpoints and parsing rules;
- difficult onboarding;
- weak reproducibility;
- fragile scaling;
- increasing maintenance cost as new sources are added.

JANUS aims to avoid that from day one by organizing the project around **explicit extraction strategies and isolated source-specific variations**.

---

## 5. Initial Scope

### In scope

- Only **federal** Brazilian public/open data.
- Only **clearly public** sources that allow public access and lawful reuse.
- 3 to 5 initial sources.
- Batch-oriented ingestion first.
- Reproducible execution on:
  - local machine;
  - single-node Spark runtime;
  - containerized environment;
  - distributed Spark cluster.

### Out of scope for the first phase

- State and municipal sources.
- Private, restricted, confidential, or non-public government data.
- Manual scraping of protected systems.
- Building a large final data warehouse for all domains.
- Full semantic unification of all public-government datasets.
- Real-time streaming as a primary mode.

---

## 6. Initial Public Federal Data Candidates

The framework should start with 3 to 5 sources that represent different access patterns. A strong initial set is:

### 6.1 Portal da Transparência (Federal Transparency Portal)
**Type:** API-oriented public federal source  
**Why include it:** represents token-based API access, filtering, pagination, and public transparency data.  
**Engineering value:** good for testing request throttling, pagination, retry, checkpointing, and incremental extraction patterns.

### 6.2 Portal Brasileiro de Dados Abertos / dados.gov.br catalog API
**Type:** metadata/catalog source  
**Why include it:** represents discovery and dataset metadata ingestion rather than direct business records.  
**Engineering value:** useful for building the source registry, metadata enrichment, and lineage catalog foundations.

### 6.3 IBGE data services
**Type:** public API source  
**Why include it:** federal statistical data with structured service interfaces.  
**Engineering value:** useful for standardized JSON/API ingestion and dimension-oriented modeling.

### 6.4 INEP microdata
**Type:** bulk file / download-based source  
**Why include it:** federal education datasets in large downloadable packages.  
**Engineering value:** useful for testing ZIP handling, schema inference, controlled ingestion zones, and large-file processing with Spark.

### 6.5 Ministry of Health open data / DATASUS open-data ecosystem
**Type:** open-data portal / bulk and tabular publication ecosystem  
**Why include it:** introduces another federal domain and heterogeneous publication style.  
**Engineering value:** useful for handling heterogeneous public-health datasets and validating extensibility.

### Criteria for accepting a source into JANUS
A source should only be accepted if it is:

- publicly available;
- legally reusable as open/public data;
- documented enough for responsible access;
- relevant for testing the framework;
- representative of a distinct ingestion pattern.

---

## 7. Architectural Principles

The architecture of JANUS must follow strong data engineering principles.

### 7.1 Metadata-driven first, but not metadata-only
The framework should rely on **metadata/configuration** to define how a source behaves whenever that behavior can be described declaratively.

Examples of configurable properties:

- source name;
- source type;
- domain;
- endpoint or download URL;
- authentication mode;
- pagination strategy;
- rate limiting rules;
- extraction mode;
- file format;
- checkpoint field;
- schema location;
- destination dataset/table/path;
- data quality rules;
- tags and ownership metadata.

However, JANUS must explicitly acknowledge that **not every source difference can or should be represented only in YAML**. Some behavior belongs in code, but it must be placed in the correct module and not leak into the whole framework.

### 7.2 Generic core, strategy-specific modules
Framework code must implement generic capabilities such as:

- HTTP client abstractions;
- file download abstractions;
- decompression;
- parsing;
- Spark read/write operations;
- partitioning;
- checkpointing;
- retries;
- logging;
- metrics;
- validation.

On top of the generic core, JANUS should have **strategy modules** for major source families, for example:

- **API extraction module**
- **file/bulk download module**
- **catalog/metadata extraction module**

Within each strategy module, JANUS may support sub-strategies such as:

- page-number pagination;
- offset pagination;
- cursor pagination;
- date-window extraction;
- static file download;
- versioned file download;
- ZIP package ingestion;
- metadata crawl patterns.

### 7.3 Configuration over duplication
Adding a new source should preferably mean:

1. create or extend a YAML file;
2. optionally create a schema definition;
3. reuse an existing strategy module;
4. add a small source-specific adapter or hook only when the source pattern is truly different.

### 7.4 Modular and layered design
The project must be organized in layers so that each layer has a clear responsibility.

### 7.5 Idempotent behavior
Rerunning the same ingestion should not create uncontrolled duplication. The framework must support deterministic reruns, checkpoint recovery, and safe overwrite/append semantics depending on the configured mode.

### 7.6 Explicit zones
The framework should use clear data layers, for example:

- **raw/landing**: exact downloaded or extracted payloads;
- **bronze**: minimally standardized Spark-readable datasets;
- **metadata**: run logs, checkpoints, lineage, source definitions, and validation results.

### 7.7 Extension by pattern, not by script copy
New source families should be added through:

- new strategy modules;
- new strategy variants;
- isolated source adapters;
- explicit contracts and tests.

They should **not** be added by copying old scripts and changing random lines.

### 7.8 Federation-ready design
The internal abstractions should make it possible to later support state and municipal sources without redesigning the whole project.

---

## 8. Reproducibility Requirements

A key requirement of JANUS is reproducibility.

### 8.1 Execution environments
The project should be executable in at least three modes:

1. **local mode** using Spark local;
2. **containerized mode** using Docker Compose or equivalent;
3. **cluster mode** using a generic Spark cluster.

### 8.2 Reproducible environment definition
The project should define:

- pinned Python version;
- pinned Spark version;
- pinned library versions;
- container image definition;
- Makefile or task runner commands;
- example environment files;
- sample local storage paths;
- optional object-storage-compatible settings.

### 8.3 No environment-specific assumptions
The code must avoid assumptions like:

- hardcoded machine paths;
- cluster-specific secrets in code;
- provider-specific infrastructure tightly coupled to business logic.

### 8.4 Infrastructure-agnostic storage contract
The framework should work with a configurable storage abstraction, such as:

- local filesystem for development;
- distributed/object storage for production-like runs.

The code should not assume only one cloud or only one storage backend.

---

## 9. Safety Requirements

JANUS must be safe both operationally and from a responsible-data-use perspective.

### 9.1 Public data only
The project must only ingest datasets that are clearly public and legally reusable.

### 9.2 No protected-system scraping
The project must not attempt to bypass authentication barriers, captchas, anti-bot protections, or restricted systems.

### 9.3 Respect source constraints
Each source integration should respect:

- documented rate limits;
- token and authentication rules when applicable;
- acceptable-use constraints;
- source update windows when relevant.

### 9.4 Secret management
If a source requires public-access tokens or keys, they must be stored outside source code, for example via:

- environment variables;
- secret files excluded from version control;
- secret managers in future deployments.

### 9.5 Safe logging
Logs must not leak:

- access tokens;
- cookies;
- sensitive headers;
- accidental personal identifiers present in payloads.

### 9.6 Responsible handling of public microdata
Even public microdata must be handled responsibly. The project must not attempt re-identification, profiling of individuals, or harmful recombination of records.

### 9.7 Safe defaults
Default behaviors should favor:

- bounded concurrency;
- retries with backoff;
- fail-fast validation of bad configs;
- explicit checkpoints;
- deterministic output paths;
- structured error reporting.

---

## 10. Non-Functional Requirements

JANUS should be evaluated not only by whether it works, but by how well it behaves.

### 10.1 Scalability
The design must support the growth from 3–5 sources to dozens of sources without architectural collapse.

### 10.2 Maintainability
The codebase must remain understandable for a new contributor.

### 10.3 Observability
The framework should provide:

- structured logs;
- run metadata;
- ingestion metrics;
- validation outcomes;
- clear failure reasons.

### 10.4 Testability
The project should have:

- unit tests for core adapters and validation logic;
- integration tests for local execution;
- sample source configs for reproducible demonstrations.

### 10.5 Portability
The project must run outside one specific cloud vendor.

### 10.6 Simplicity
The framework must stay as simple as possible. It should solve the current problem without overengineering, while still leaving room for growth.

---

## 11. Recommended High-Level Architecture

A clean high-level architecture for JANUS is:

1. **Source Registry**  
   YAML files that describe the sources and map them to an extraction strategy.

2. **Planner**  
   Reads source metadata and decides how the ingestion job should run.

3. **Strategy Layer**  
   Chooses the correct execution family, for example:
   - API strategy
   - bulk file strategy
   - catalog strategy

4. **Strategy Modules**  
   Each strategy module contains reusable logic for that family.

   Example:
   - `api/` for HTTP-based sources
   - `files/` for CSV/ZIP/XLSX downloads
   - `catalog/` for metadata-oriented sources

5. **Strategy Variants / Components**  
   Reusable subcomponents inside a strategy, for example:
   - page paginator
   - offset paginator
   - date-window iterator
   - archive extractor
   - file resolver
   - response flattener

6. **Source-Specific Adapters or Hooks**  
   Small isolated modules for the cases where a source differs from the standard behavior of its family.

7. **Normalizer**  
   Converts raw payloads into minimally standardized datasets.

8. **Spark Processing Layer**  
   Reads, partitions, transforms, validates, and writes datasets.

9. **Checkpoint & Lineage Store**  
   Stores execution state, source version information, and run metadata.

10. **Output Zones**  
    Persists raw, bronze, and metadata outputs.

### Architectural rule
The control flow should be:

**YAML -> planner -> strategy module -> optional source-specific hook -> Spark normalization/write**

This keeps YAML useful without pretending that YAML alone is enough.

---

## 12. Strategy-Based Design Model

This is one of the most important design decisions in JANUS.

### 12.1 Why strategy modules are necessary
A fully generic source-agnostic model is desirable at the architectural level, but unrealistic at the implementation level. Public data sources differ too much in practice.

Therefore, JANUS should adopt the following principle:

**The project is agnostic in its foundation, but modular in its execution.**

That means:

- shared concerns stay in common modules;
- behavior common to a source family stays in a strategy module;
- behavior unique to one source stays in a source-specific adapter.

### 12.2 Recommended strategy families

#### API strategy module
Responsible for sources that expose HTTP APIs.

Typical reusable responsibilities:
- HTTP client lifecycle
- auth/header injection
- retries and backoff
- pagination execution
- response persistence to raw zone
- checkpoint-based incremental extraction
- response-to-record normalization hooks

Possible subtypes:
- page-based API
- offset-based API
- cursor-based API
- date-window API
- mixed endpoint API

#### File strategy module
Responsible for sources that publish files for download.

Typical reusable responsibilities:
- file discovery
- version resolution
- download management
- checksum or file integrity checks
- archive extraction
- file format detection
- handoff to Spark readers

Possible subtypes:
- static URL file
- periodic file publication
- versioned package source
- multi-file archive source

#### Catalog strategy module
Responsible for metadata-first sources.

Typical reusable responsibilities:
- dataset catalog traversal
- organization/group/resource extraction
- metadata normalization
- lineage-friendly persistence

### 12.3 Source-specific adapters
A source-specific adapter should exist only when a source cannot be expressed cleanly through an existing strategy plus configuration.

Examples:
- custom pagination semantics;
- irregular nested payload flattening;
- source-specific request signing;
- special file-name resolution logic;
- source-specific post-processing before Spark normalization.

### 12.4 Anti-spaghetti rule
Source-specific adapters must be:

- small;
- isolated;
- well-named;
- tested;
- clearly associated with one source.

They must not contain unrelated framework concerns.

---

## 13. Suggested Repository Structure

```text
janus/
├── README.md
├── pyproject.toml
├── requirements.txt
├── Makefile
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── conf/
│   ├── app.yaml
│   ├── environments/
│   │   ├── local.yaml
│   │   └── cluster.yaml
│   ├── sources/
│   │   ├── transparencia/
│   │   ├── dados_abertos_catalog/
│   │   ├── receita_federal/
│   │   ├── ibge/
│   │   ├── inep/
│   │   └── example/
│   └── schemas/
│       ├── transparencia/
│       ├── receita_federal/
│       ├── ibge/
│       └── inep/
├── src/
│   └── janus/
│       ├── schema_contracts.py
│       ├── main.py
│       ├── scripts/
│       ├── planner/
│       ├── registry/
│       ├── strategies/
│       │   ├── base/
│       │   ├── api/
│       │   ├── files/
│       │   └── catalog/
│       ├── hooks/
│       │   ├── transparencia/
│       │   ├── ibge/
│       │   └── inep/
│       ├── readers/
│       ├── normalizers/
│       ├── writers/
│       ├── quality/
│       ├── checkpoints/
│       ├── lineage/
│       ├── utils/
│       └── models/
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── data/
│   ├── raw/
│   ├── bronze/
│   └── metadata/
└── docs/
    ├── architecture.md
    ├── source-onboarding.md
    ├── strategy-patterns.md
    └── reproducibility.md
```

This structure helps keep the framework navigable as features grow.

---

## 14. YAML-Driven Source Definition

One of the main best practices in JANUS is to define sources declaratively whenever possible.

### Example source YAML

```yaml
source_id: transparencia__poder_executivo_federal__servidores_por_orgao__full_refresh
name: Portal da Transparencia - Servidores agregados por orgao
owner: janus
enabled: false
source_type: api
strategy: api
strategy_variant: page_number_api
federation_level: federal
domain: transparencia
public_access: true

access:
  base_url: https://api.portaldatransparencia.gov.br
  path: /api-de-dados/servidores/por-orgao
  method: GET
  format: json
  timeout_seconds: 60
  headers:
    Accept: application/json
  auth:
    type: header_token
    env_var: TRANSPARENCIA_API_TOKEN
    header_name: chave-api-dados
  pagination:
    type: page_number
    page_param: pagina
    size_param: tamanhoPagina
    page_size: 15
  rate_limit:
    requests_per_minute: 90
    concurrency: 10
    backoff_seconds: 10

extraction:
  mode: full_refresh
  checkpoint_strategy: none
  dead_letter_max_items: 100
  retry:
    max_attempts: 4
    backoff_strategy: exponential
    backoff_seconds: 5

schema:
  contract: conf/contracts/transparencia/poder_executivo_federal__servidores_por_orgao.yaml

spark:
  input_format: json
  write_mode: overwrite
  repartition: 1
  partition_by:
    - ingestion_date

outputs:
  raw:
    path: data/raw/transparencia/poder_executivo_federal__servidores_por_orgao
    format: json
  bronze:
    path: data/bronze/transparencia/poder_executivo_federal__servidores_por_orgao
    format: iceberg
    namespace: bronze__transparencia
    table_name: poder_executivo_federal__servidores_por_orgao
  metadata:
    path: data/metadata/transparencia/poder_executivo_federal__servidores_por_orgao
    format: json

quality:
  required_fields:
    - qntPessoas
    - qntVinculos
    - codOrgaoExercicioSiape
  allow_schema_evolution: true
```

### Why YAML matters
Using YAML for source onboarding gives the project:

- clearer contracts;
- lower onboarding cost;
- less hardcoded logic;
- easier review of source definitions;
- a path to future automation.

In the current JANUS source contract, output settings live under `outputs`, with one target each for `raw`, `bronze`, and `metadata`. Each target defines its own `path` and `format`. When `outputs.bronze.format` is `iceberg`, the bronze target may also define `namespace` and `table_name` to override the default path-derived Iceberg identifier.

### What YAML should and should not do
YAML **should** define:

- which strategy to use;
- which variant to use;
- configurable request or file parameters;
- checkpoint semantics;
- output contracts, including per-zone paths and formats and optional `outputs.bronze.namespace` plus `outputs.bronze.table_name` for Iceberg bronze outputs;
- quality rules.

YAML **should not** try to encode all business logic or every custom behavior. If the source requires special treatment, that logic belongs in a strategy extension or source hook.

---

## 15. Spark Usage Principles

Spark should be used intentionally, not only because it is a popular tool.

### JANUS should use Spark for:

- large-file ingestion;
- standardized parsing of heterogeneous payloads;
- distributed transformation when volume justifies it;
- partitioned writes;
- scalable normalization;
- reproducible execution across environments.

### JANUS should avoid using Spark for:

- tiny control logic that is simpler in plain Python;
- unnecessary distributed complexity for trivial tasks;
- tightly coupling source-fetch logic to Spark internals.

### Good Spark practices for JANUS

- isolate Spark session creation in one module;
- keep Spark configs environment-driven;
- use explicit schemas whenever feasible;
- control partitioning deliberately;
- avoid hidden side effects inside transformations;
- write deterministic outputs;
- make local mode and cluster mode behavior as similar as possible.

---

## 16. Data Engineering Best Practices to Enforce

JANUS should explicitly follow these practices:

### 16.1 Separation of concerns
Extraction, normalization, storage, validation, and orchestration must be separated.

### 16.2 Configuration-driven pipelines
Source behavior should be described declaratively where possible.

### 16.3 Explicit contracts
Schemas, checkpoint semantics, write modes, strategy families, and hook points must be explicit.

### 16.4 Idempotency and rerunnability
Reruns must be safe and predictable.

### 16.5 Versioned configuration
Source definitions and schema files must be version-controlled.

### 16.6 Small interfaces
Each strategy and adapter should implement a minimal contract.

### 16.7 Pattern-based extensibility
New features should be added through strategy modules, variants, and isolated hooks instead of random patches.

### 16.8 Documentation-first onboarding
Adding a new source should follow a clear onboarding guide.

### 16.9 Prefer composition over condition explosion
The framework should avoid giant `if/else` chains for source behavior. Reusable components should be composed by strategy and variant.

---

## 17. How JANUS Avoids Becoming a Frankenstein Project

This is one of the central design concerns of JANUS.

To avoid becoming a Frankenstein or spaghetti-code project, JANUS must enforce:

- a stable folder structure;
- source configs outside core logic;
- standard strategy interfaces;
- source hooks isolated by source;
- naming conventions;
- run metadata standards;
- clear ownership of each module;
- mandatory tests for new strategy variants and new hooks;
- architecture notes for every new pattern introduced.

### Rule of thumb
If onboarding a new source requires copying an old script and editing it by hand, the framework design is failing.

A new source should preferably reuse:

- one existing strategy;
- one strategy variant;
- one config schema;
- one output contract;
- one checkpoint pattern.

Only when necessary should it add:

- one small source-specific hook.

---

## 18. Suggested Initial Research Focus for the Post-Graduation Work

The post-graduation work should not try to solve every possible problem. A strong first focus is:

### Research focus
**How can a metadata-driven and strategy-based Spark framework ingest heterogeneous federal Brazilian public data sources in a reproducible, safe, and scalable way?**

### Suggested evaluation dimensions
- reproducibility;
- extensibility;
- ingestion throughput;
- rerun safety;
- recoverability after failure;
- onboarding effort for a new source;
- architectural cleanliness;
- degree of strategy reuse across sources.

---

## 19. Suggested Deliverables

The first academic/project phase should deliver:

1. A working JANUS framework.
2. Source configs for 3 to 5 federal public data sources.
3. At least 2 to 3 reusable strategy modules.
4. Reproducible local and cluster execution instructions.
5. A documented architecture.
6. Basic data quality checks.
7. Run metadata and checkpoint support.
8. An evaluation section showing how the framework behaves.

---

## 20. Future Evolution Path

### Next project stage
Expand from federal-only to:

- state-level public data;
- municipal public data.

### Later technical evolution
- broader source-type support;
- better schema evolution handling;
- adaptive extraction strategies;
- stronger lineage and provenance tracking;
- orchestration integration;
- benchmark suite for heterogeneous public ingestion.

### Academic evolution
- **Current phase:** architecture and implementation for post-graduation final work.
- **Master's:** adaptive ingestion strategy and source classification.
- **PhD:** distributed optimization, self-healing pipelines, and resilient heterogeneous data ingestion.

---

## 21. Proposed Working Title

**JANUS: A Metadata-Driven and Strategy-Based Spark Framework for Reproducible and Safe Ingestion of Brazilian Federal Open Data**

---

## 22. Final Statement

JANUS is not a one-off collection of scripts.

It is a deliberate attempt to build a **clean**, **extensible**, **Spark-oriented**, and **federation-ready** data engineering foundation for public government data ingestion. The first version will intentionally stay limited to a few **federal public sources**, but the architecture must already be good enough to grow later without collapse.

Its key architectural stance is realistic:

- the foundation should be agnostic and reusable;
- the execution should be organized by source strategy;
- YAML should drive configuration;
- source-specific behavior should exist only when necessary and stay isolated.

If JANUS succeeds in this first phase, it will provide:

- a strong post-graduation final project;
- a clear path for future academic work;
- a practical demonstration of good data engineering architecture applied to real public data.

---

## 23. Official Reference Sources for the Initial Federal Scope

The initial source candidates above are aligned with official federal public-data endpoints and portals, including:

- Portal Brasileiro de Dados Abertos API / dados.gov.br catalog
- Portal da Transparência API
- IBGE data APIs
- INEP open-data and microdata pages
- Ministry of Health / DATASUS open-data portals

These references should be reviewed during implementation to confirm endpoint details, authentication requirements, and acceptable usage constraints.
