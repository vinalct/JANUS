COMPOSE_FILE := docker/docker-compose.yml
PODMAN_COMPOSE_FILE := docker/docker-compose.podman.yml
SERVICE := janus
ENVIRONMENT ?= local
RUN_ARGS ?=
PYTHON ?= python
JANUS_UID := $(shell id -u)
JANUS_GID := $(shell id -g)
JANUS_PROJECT_ROOT := $(CURDIR)
COMMA := ,

ICEBERG_JAR_NAME := org.apache.iceberg_iceberg-spark-runtime-4.0_2.13-1.10.1.jar
SQLITE_JAR_VERSION := 3.53.2.1
SQLITE_JAR_NAME := org.xerial_sqlite-jdbc-$(SQLITE_JAR_VERSION).jar
SQLITE_DRIVER_PACKAGE := org.xerial:sqlite-jdbc:$(SQLITE_JAR_VERSION)
POSTGRES_JAR_VERSION := 42.7.13
POSTGRES_JAR_NAME := org.postgresql_postgresql-$(POSTGRES_JAR_VERSION).jar
IVY_JAR_NAMES := $(ICEBERG_JAR_NAME) $(SQLITE_JAR_NAME) $(POSTGRES_JAR_NAME)
IVY_JAR_DEST_DIR := data/metadata/ivy/jars

DAGSTER_VERSION := 1.13.22
DAGSTER_TEST_REPORT := data/metadata/test-reports/dagster.xml
FAST_TEST_REPORT := data/metadata/test-reports/fast.xml
CI_TEST_REPORT := data/metadata/test-reports/ci.xml
DAGSTER_TESTS := tests/unit/adapters/test_dagster_adapter.py \
	tests/integration/dagster \
	tests/unit/examples/test_orchestration_example.py::test_dagster_definitions_have_exact_edge_and_disabled_schedule
FAST_TEST_ARGS := tests/unit \
	--ignore=tests/unit/adapters/test_dagster_adapter.py \
	--deselect=tests/unit/examples/test_orchestration_example.py::test_dagster_definitions_have_exact_edge_and_disabled_schedule
CONTRACT_SCHEMA_TESTS := \
	tests/unit/models/data_contracts/test_contract_loader.py::test_pinned_odcs_schema_sha256_matches_the_sidecar \
	tests/unit/models/data_contracts/test_contract_loader.py::test_every_checked_in_contract_validates_against_the_pinned_odcs_schema \
	tests/unit/models/data_contracts/test_contract_loader.py::test_odcs_validator_rejects_the_hostile_fixtures
FAST_OBSERVABILITY_CLASSES := \
	tests.unit.observability.test_acceptance_evidence \
	tests.unit.observability.test_architecture_guardrails \
	tests.unit.observability.test_example_queries_and_docs \
	tests.unit.observability.test_failure_injection_matrix \
	tests.unit.observability.test_guarded_emission \
	tests.unit.observability.test_openlineage_facets \
	tests.unit.observability.test_openlineage_mapping_failure \
	tests.unit.observability.test_openlineage_transports \
	tests.unit.observability.test_profile_spark_isolation \
	tests.unit.observability.test_pyiceberg_append_sink \
	tests.unit.observability.test_run_record_projection \
	tests.unit.observability.test_runs_table_contract
GUARDRAIL_CLASSES := \
	tests.unit.strategies.http.test_transport_boundary \
	tests.unit.strategies.http.test_no_second_opener \
	tests.unit.strategies.http.test_secret_scrubber \
	tests.unit.strategies.files.test_hostile_listing \
	tests.unit.strategies.files.test_payload_and_archive_limits \
	tests.unit.writers.test_write_stream \
	tests.unit.utils.test_redaction_gaps \
	tests.unit.utils.test_runtime_fallback \
	tests.unit.checkpoints.test_progress_prefix_containment \
	tests.unit.models.test_access_limits_and_allowed_hosts \
	tests.unit.toolchain.test_container_hardening \
	tests.unit.toolchain.test_ci_supply_chain
CONTRACT_ENFORCEMENT_CLASSES := \
	tests.unit.quality.test_contract_checks \
	tests.unit.runtime.test_pre_write_gate_wiring \
	tests.unit.writers.test_evolution_plan \
	tests.unit.writers.test_schema_ddl \
	tests.unit.runtime.test_contract_preflight \
	tests.unit.toolchain.test_no_write_without_contract_check \
	tests.unit.models.test_schema_config_contract_only
SPARK_ORCHESTRATION_CLASS := tests.integration.orchestration.test_dependency_execution
QUERYABLE_CLASS := tests.integration.catalog_commits.test_queryable_observability
RUNS_SINK_CLASS := tests.integration.catalog_commits.test_runs_table_append_sink
RUNS_DECLARATION_CLASS := tests.unit.observability.test_pyiceberg_declared_schema
PREFLIGHT_REAL_CATALOG_CLASS := tests.integration.catalog_commits.test_contract_preflight_real_catalog
CROSS_ENGINE_CLASS := tests.integration.catalog_commits.test_pyiceberg_round_trip
CONTRACTS_GOLDEN_CLASS := tests.integration.contracts.test_bronze_unchanged_after_migration
PRE_WRITE_GATE_CLASS := tests.integration.contracts.test_pre_write_gate
MALFORMED_ROWS_CLASS := tests.integration.contracts.test_malformed_rows
EVOLUTION_MATRIX_CLASS := tests.integration.contracts.test_evolution_matrix
AC1_TEST := test_real_terminal_runs_land_field_by_field_and_spark_reads_across_sources
AC2_TEST := test_published_ac2_queries_execute_verbatim_with_retry_and_window_boundaries
CROSS_ENGINE_TEST := test_spark_reads_the_row_pyiceberg_committed

# ── the cluster profile ─────────────────────────────────────────────────────
CLUSTER_PROFILE := cluster
CLUSTER_ENV_FILE := conf/environments/cluster.env
CLUSTER_SERVICE := janus-cluster
CLUSTER_REST_PROFILE := cluster-rest
CLUSTER_REST_SERVICE := janus-cluster-rest
AWS_BUNDLE_VERSION := 1.10.1
AWS_BUNDLE_JAR_NAME := org.apache.iceberg_iceberg-aws-bundle-$(AWS_BUNDLE_VERSION).jar
AWS_BUNDLE_URL := https://repo1.maven.org/maven2/org/apache/iceberg/iceberg-aws-bundle/$(AWS_BUNDLE_VERSION)/iceberg-aws-bundle-$(AWS_BUNDLE_VERSION).jar
AWS_BUNDLE_SHA256 := 86bf20892ea5b4c17688f19b075399885f6aa5303f6b2dc9f491e76ceef9633b

# The catalog the local profile points at, and the defaults `pyspark-local` falls back to so
# the REPL and the app open the same one. Keep in lockstep with conf/environments/local.yaml.
ICEBERG_CATALOG_DIR := data/metadata/iceberg-catalog
ICEBERG_CATALOG_URI := jdbc:sqlite:$(ICEBERG_CATALOG_DIR)/catalog.sqlite?journal_mode=WAL&busy_timeout=30000
ICEBERG_JDBC_SCHEMA_VERSION := V1

# Runs in the compose shell before the engine is invoked; the cluster targets use it to
# export the generated stack credentials so compose can substitute them. Default: no-op.
COMPOSE_PRELUDE ?= :

DETECT_COMPOSE = if podman compose version >/dev/null 2>&1; then echo 'podman compose'; elif command -v podman-compose >/dev/null 2>&1; then echo podman-compose; elif docker compose version >/dev/null 2>&1; then echo 'docker compose'; elif command -v docker-compose >/dev/null 2>&1; then echo docker-compose; else exit 1; fi

define RUN_COMPOSE
	@compose_cmd="$$( $(DETECT_COMPOSE) )" || { \
		echo "No compose-capable container engine found. Install Docker Compose, docker-compose, podman compose, or podman-compose." >&2; \
		exit 1; \
	}; \
	compose_files="$$(case "$$compose_cmd" in podman* ) printf '%s' '-f $(COMPOSE_FILE) -f $(PODMAN_COMPOSE_FILE)' ;; * ) printf '%s' '-f $(COMPOSE_FILE)' ;; esac)"; \
	container_user="$$(case "$$compose_cmd" in podman* ) printf '%s:%s' '$(JANUS_UID)' '$(JANUS_GID)' ;; * ) printf '%s:%s' '$(JANUS_UID)' '$(JANUS_GID)' ;; esac)"; \
	$(COMPOSE_PRELUDE); \
	JANUS_CONTAINER_USER=$$container_user JANUS_UID=$(JANUS_UID) JANUS_GID=$(JANUS_GID) JANUS_PROJECT_ROOT=$(JANUS_PROJECT_ROOT) $$compose_cmd $$compose_files $(1)
endef

.PHONY: bootstrap check-compose check-env up ensure-up seed-ivy down status logs shell pyspark-local lint typecheck test test-fast test-contract-schema test-adapter ci run-local run-local-config docker-build docker-run clean cluster-secrets seed-cluster-jars up-cluster down-cluster status-cluster logs-cluster shell-cluster run-cluster test-cluster up-cluster-rest down-cluster-rest status-cluster-rest logs-cluster-rest shell-cluster-rest run-cluster-rest test-cluster-rest

seed-ivy:
	@mkdir -p "$(IVY_JAR_DEST_DIR)" "$(ICEBERG_CATALOG_DIR)"; \
	for jar in $(IVY_JAR_NAMES); do \
		if [ ! -f "$(IVY_JAR_DEST_DIR)/$$jar" ]; then \
			echo "Seeding $$jar from deps/"; \
			cp "deps/$$jar" "$(IVY_JAR_DEST_DIR)/$$jar"; \
		fi; \
	done

check-compose:
	@compose_cmd="$$( $(DETECT_COMPOSE) )" || { \
		echo "No compose-capable container engine found. Install Docker Compose, docker-compose, podman compose, or podman-compose." >&2; \
		exit 1; \
	}; \
	printf 'Using %s\n' "$$compose_cmd"; \
	case "$$compose_cmd" in podman* ) printf 'Using Podman keep-id user namespace for writable bind mounts\n' ;; esac

# Warn when the local secrets file is readable by group/other. A warning, not an error:
# the file is optional and CI has none.
check-env:
	@if [ -f .env ] && [ -n "$$(find .env -maxdepth 0 -perm /077 2>/dev/null)" ]; then \
		echo "warning: .env is readable by group/other — run: chmod 600 .env" >&2; \
	fi

bootstrap: check-compose check-env
	$(call RUN_COMPOSE,build $(SERVICE))

# Explicit fresh restart — stops and recreates the container.
up: check-compose check-env seed-ivy
	$(call RUN_COMPOSE,up -d --force-recreate $(SERVICE))

ensure-up: check-compose check-env seed-ivy
	$(call RUN_COMPOSE,up -d $(SERVICE))

down: check-compose
	$(call RUN_COMPOSE,down)

status: check-compose
	$(call RUN_COMPOSE,ps)

logs: check-compose
	$(call RUN_COMPOSE,logs $(SERVICE))

shell: ensure-up
	$(call RUN_COMPOSE,exec $(SERVICE) sh)

# The REPL reads the same catalog knobs the environment profile does
# (conf/environments/*.yaml) so `make pyspark-local` and the app cannot drift onto
# different catalogs. The `jdbc` default here is the profile's default, and moves
# with it. The project-relative path inside a file-backed URI is resolved the same way
# the directories below are, because the app resolves it too (build_spark_options) and a
# relative path would otherwise follow the REPL's working directory.
pyspark-local: ensure-up
	$(call RUN_COMPOSE,exec $(SERVICE) sh -lc '\
	ivy_dir="$${JANUS_SPARK_IVY_DIR:-data/metadata/ivy}"; \
	if [ "$$ivy_dir" = "$${ivy_dir#/}" ]; then ivy_dir="/workspace/$$ivy_dir"; fi; \
	iceberg_warehouse="$${JANUS_ICEBERG_WAREHOUSE_DIR:-data/bronze/iceberg}"; \
	if [ "$$iceberg_warehouse" = "$${iceberg_warehouse#/}" ]; then iceberg_warehouse="/workspace/$$iceberg_warehouse"; fi; \
	spark_warehouse="$${JANUS_SPARK_WAREHOUSE_DIR:-data/metadata/spark-warehouse}"; \
	if [ "$$spark_warehouse" = "$${spark_warehouse#/}" ]; then spark_warehouse="/workspace/$$spark_warehouse"; fi; \
	catalog="$${JANUS_ICEBERG_CATALOG_NAME:-janus}"; \
	catalog_type="$${JANUS_ICEBERG_CATALOG_TYPE:-jdbc}"; \
	packages="$${JANUS_ICEBERG_RUNTIME_PACKAGE:-org.apache.iceberg:iceberg-spark-runtime-4.0_2.13:1.10.1}"; \
	catalog_uri="$${JANUS_ICEBERG_CATALOG_URI:-}"; \
	if [ "$$catalog_type" = jdbc ] && [ -z "$$catalog_uri" ]; then catalog_uri="$(ICEBERG_CATALOG_URI)"; fi; \
	uri_path="$${catalog_uri#jdbc:sqlite:}"; \
	if [ "$$uri_path" != "$$catalog_uri" ] && [ "$$uri_path" = "$${uri_path#/}" ]; then catalog_uri="jdbc:sqlite:/workspace/$$uri_path"; fi; \
	catalog_conf="--conf spark.sql.catalog.$$catalog.type=$$catalog_type"; \
	if [ "$$catalog_type" = jdbc ] || [ "$$catalog_type" = rest ]; then \
		if [ -z "$$catalog_uri" ]; then \
			echo "JANUS_ICEBERG_CATALOG_URI must be set for catalog type $$catalog_type" >&2; \
			exit 1; \
		fi; \
		catalog_conf="$$catalog_conf --conf spark.sql.catalog.$$catalog.uri=$$catalog_uri"; \
	fi; \
	if [ "$$catalog_type" = jdbc ]; then \
		catalog_conf="$$catalog_conf --conf spark.sql.catalog.$$catalog.jdbc.schema-version=$(ICEBERG_JDBC_SCHEMA_VERSION)"; \
		driver_package="$${JANUS_ICEBERG_JDBC_DRIVER_PACKAGE:-$(SQLITE_DRIVER_PACKAGE)}"; \
		if [ -n "$$driver_package" ]; then packages="$$packages$(COMMA)$$driver_package"; fi; \
		if [ -n "$${JANUS_ICEBERG_CATALOG_USER:-}" ]; then catalog_conf="$$catalog_conf --conf spark.sql.catalog.$$catalog.jdbc.user=$$JANUS_ICEBERG_CATALOG_USER"; fi; \
		if [ -n "$${JANUS_ICEBERG_CATALOG_PASSWORD:-}" ]; then catalog_conf="$$catalog_conf --conf spark.sql.catalog.$$catalog.jdbc.password=$$JANUS_ICEBERG_CATALOG_PASSWORD"; fi; \
	fi; \
	pyspark \
		--packages "$$packages" \
		--conf spark.jars.ivy="$$ivy_dir" \
		--conf spark.sql.extensions=org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions \
		--conf spark.sql.defaultCatalog="$$catalog" \
		--conf spark.sql.catalog.$$catalog=org.apache.iceberg.spark.SparkCatalog \
		$$catalog_conf \
		--conf spark.sql.catalog.$$catalog.warehouse="$$iceberg_warehouse" \
		--conf spark.sql.catalog.$$catalog.default-namespace="$${JANUS_ICEBERG_DEFAULT_NAMESPACE:-bronze}" \
		--conf spark.sql.warehouse.dir="$$spark_warehouse" \
		--conf spark.driver.bindAddress="$${JANUS_SPARK_DRIVER_BIND_ADDRESS:-127.0.0.1}" \
		--conf spark.driver.host="$${JANUS_SPARK_DRIVER_HOST:-127.0.0.1}" \
		--conf spark.sql.session.timeZone=UTC \
		--conf spark.ui.enabled=false')

lint: ensure-up
	$(call RUN_COMPOSE,exec -T $(SERVICE) python -m ruff check src tests)

typecheck: ensure-up
	$(call RUN_COMPOSE,exec -T $(SERVICE) python -m mypy)

test: ensure-up
	$(call RUN_COMPOSE,exec -T $(SERVICE) python -m pytest)

# Dependency-free orchestration and repository guardrails, matching CI's host-side fast job.
test-fast:
	@mkdir -p "$(dir $(FAST_TEST_REPORT))"
	$(PYTHON) -m pytest -ra $(FAST_TEST_ARGS) --junitxml="$(FAST_TEST_REPORT)"
	@for class_name in $(FAST_OBSERVABILITY_CLASSES); do \
		$(PYTHON) -m tests.support.required_test_gate "$(FAST_TEST_REPORT)" \
			--class-name "$$class_name" --minimum-passed 1 || exit $$?; \
	done
	@for class_name in $(GUARDRAIL_CLASSES); do \
		$(PYTHON) -m tests.support.required_test_gate "$(FAST_TEST_REPORT)" \
			--class-name "$$class_name" --minimum-passed 1 || exit $$?; \
	done
	@for class_name in $(CONTRACT_ENFORCEMENT_CLASSES); do \
		$(PYTHON) -m tests.support.required_test_gate "$(FAST_TEST_REPORT)" \
			--class-name "$$class_name" --minimum-passed 1 || exit $$?; \
	done

test-contract-schema:
	$(PYTHON) -m pytest -q $(CONTRACT_SCHEMA_TESTS)

test-adapter:
	@mkdir -p "$(dir $(DAGSTER_TEST_REPORT))"
	$(PYTHON) -c "import dagster; from importlib.metadata import version; actual = version('dagster'); assert actual == '$(DAGSTER_VERSION)', f'expected dagster $(DAGSTER_VERSION), found {actual}'"
	$(PYTHON) -m pytest -ra $(DAGSTER_TESTS) --junitxml="$(DAGSTER_TEST_REPORT)"
	$(PYTHON) -m tests.support.required_test_gate "$(DAGSTER_TEST_REPORT)" --minimum-passed 13

# Reproduce CI locally: same lint + type check + full suite the container CI job runs,
# in the container so the Spark/Iceberg path is exercised. Keep in lockstep with .github/workflows/ci.yml.
ci: ensure-up
	$(call RUN_COMPOSE,exec -T $(SERVICE) python -m ruff check src tests)
	$(call RUN_COMPOSE,exec -T $(SERVICE) python -m mypy)
	$(call RUN_COMPOSE,exec -T -e COVERAGE_FILE=data/.coverage $(SERVICE) python -m pytest -ra -m 'not cluster' --cov=janus --cov-report=term-missing --cov-report=xml:data/coverage.xml --junitxml=$(CI_TEST_REPORT))
	$(call RUN_COMPOSE,exec -T $(SERVICE) python -m tests.support.required_test_gate $(CI_TEST_REPORT) --class-name $(SPARK_ORCHESTRATION_CLASS) --minimum-passed 8)
	$(call RUN_COMPOSE,exec -T $(SERVICE) python -m tests.support.required_test_gate $(CI_TEST_REPORT) --class-name $(RUNS_SINK_CLASS) --minimum-passed 4)
	$(call RUN_COMPOSE,exec -T $(SERVICE) python -m tests.support.required_test_gate $(CI_TEST_REPORT) --class-name $(RUNS_DECLARATION_CLASS) --minimum-passed 1)
	$(call RUN_COMPOSE,exec -T $(SERVICE) python -m tests.support.required_test_gate $(CI_TEST_REPORT) --class-name $(PREFLIGHT_REAL_CATALOG_CLASS) --minimum-passed 10)
	$(call RUN_COMPOSE,exec -T $(SERVICE) python -m tests.support.required_test_gate $(CI_TEST_REPORT) --class-name $(PRE_WRITE_GATE_CLASS) --minimum-passed 13)
	$(call RUN_COMPOSE,exec -T $(SERVICE) python -m tests.support.required_test_gate $(CI_TEST_REPORT) --class-name $(MALFORMED_ROWS_CLASS) --minimum-passed 9)
	$(call RUN_COMPOSE,exec -T $(SERVICE) python -m tests.support.required_test_gate $(CI_TEST_REPORT) --class-name $(EVOLUTION_MATRIX_CLASS) --minimum-passed 59)
	$(call RUN_COMPOSE,exec -T $(SERVICE) python -m tests.support.required_test_gate $(CI_TEST_REPORT) --class-name $(QUERYABLE_CLASS) --test-name $(AC1_TEST) --minimum-passed 1)
	$(call RUN_COMPOSE,exec -T $(SERVICE) python -m tests.support.required_test_gate $(CI_TEST_REPORT) --class-name $(QUERYABLE_CLASS) --test-name $(AC2_TEST) --minimum-passed 1)
	$(call RUN_COMPOSE,exec -T $(SERVICE) python -m tests.support.required_test_gate $(CI_TEST_REPORT) --class-name $(CROSS_ENGINE_CLASS) --test-name $(CROSS_ENGINE_TEST) --minimum-passed 1)
	$(call RUN_COMPOSE,exec -T $(SERVICE) python -m tests.support.required_test_gate $(CI_TEST_REPORT) --class-name $(CONTRACTS_GOLDEN_CLASS) --minimum-passed 23)

run-local: ensure-up
	$(call RUN_COMPOSE,exec -T $(SERVICE) python -m janus.main --environment $(ENVIRONMENT) --with-spark)

run-local-config: ensure-up
	$(call RUN_COMPOSE,exec -T $(SERVICE) python -m janus.main --environment $(ENVIRONMENT))

# ── cluster profile: MinIO + Postgres + S3FileIO ────────────────────────────
# Opt-in by construction: every target below passes `--profile cluster`, so `make up`
# remains a one-container experience.

# Per-machine credentials for the stack, generated once into an untracked file. No
# password is ever committed, and rotating is `rm conf/environments/cluster.env`.
cluster-secrets:
	@if [ -f "$(CLUSTER_ENV_FILE)" ]; then exit 0; fi; \
	if [ ! -r /dev/urandom ]; then \
		echo "/dev/urandom is unavailable; write $(CLUSTER_ENV_FILE) by hand." >&2; \
		exit 1; \
	fi; \
	secret() { head -c 24 /dev/urandom | od -An -tx1 | tr -d ' \n'; }; \
	umask 077; \
	{ \
		echo "# Generated by \`make cluster-secrets\`. Untracked (.gitignore) and local to this"; \
		echo "# machine: the cluster stack's own credentials, nothing JANUS itself configures."; \
		echo "# Delete this file and re-run \`make up-cluster\` to rotate them."; \
		echo "JANUS_CLUSTER_POSTGRES_USER=janus"; \
		echo "JANUS_CLUSTER_POSTGRES_DB=iceberg"; \
		echo "JANUS_CLUSTER_POSTGRES_PASSWORD=$$(secret)"; \
		echo "JANUS_CLUSTER_S3_ACCESS_KEY=janus"; \
		echo "JANUS_CLUSTER_S3_SECRET_KEY=$$(secret)"; \
	} > "$(CLUSTER_ENV_FILE)"; \
	echo "Generated $(CLUSTER_ENV_FILE)"

seed-cluster-jars:
	@mkdir -p "$(IVY_JAR_DEST_DIR)"; \
	jar="$(IVY_JAR_DEST_DIR)/$(AWS_BUNDLE_JAR_NAME)"; \
	if [ ! -f "$$jar" ]; then \
		echo "Fetching $(AWS_BUNDLE_JAR_NAME) (59.8 MiB, once)"; \
		if command -v curl >/dev/null 2>&1; then curl -fsSL -o "$$jar.part" "$(AWS_BUNDLE_URL)"; \
		elif command -v wget >/dev/null 2>&1; then wget -q -O "$$jar.part" "$(AWS_BUNDLE_URL)"; \
		else echo "Neither curl nor wget is available; download $(AWS_BUNDLE_URL) into $$jar" >&2; exit 1; \
		fi; \
		mv "$$jar.part" "$$jar"; \
	fi; \
	echo "$(AWS_BUNDLE_SHA256)  $$jar" | sha256sum -c - || { \
		echo "$(AWS_BUNDLE_JAR_NAME) failed verification; removing it." >&2; rm -f "$$jar"; exit 1; \
	}

up-cluster down-cluster status-cluster logs-cluster shell-cluster run-cluster test-cluster \
up-cluster-rest down-cluster-rest status-cluster-rest logs-cluster-rest shell-cluster-rest run-cluster-rest test-cluster-rest: \
	COMPOSE_PRELUDE = set -a; [ -f ./$(CLUSTER_ENV_FILE) ] && . ./$(CLUSTER_ENV_FILE); set +a

up-cluster: check-compose seed-ivy seed-cluster-jars cluster-secrets
	$(call RUN_COMPOSE,--profile $(CLUSTER_PROFILE) up -d)

# Stops and removes the stack's containers. The named volumes survive on purpose — the
# catalog rows and the bronze objects must outlive the containers holding them, which is
# what the restart-survival evidence turns on. Wipe them with `down-cluster` plus
# `<engine> volume rm janus-postgres-data janus-minio-data`.
down-cluster: check-compose
	$(call RUN_COMPOSE,--profile $(CLUSTER_PROFILE) down)

status-cluster: check-compose
	$(call RUN_COMPOSE,--profile $(CLUSTER_PROFILE) ps -a)

logs-cluster: check-compose
	$(call RUN_COMPOSE,--profile $(CLUSTER_PROFILE) logs --no-color postgres minio minio-init)

shell-cluster: up-cluster
	$(call RUN_COMPOSE,--profile $(CLUSTER_PROFILE) exec $(CLUSTER_SERVICE) sh)

# Mirrors run-local, against the cluster profile. Extra CLI arguments go in RUN_ARGS, e.g.
# `make run-cluster RUN_ARGS="--source-id <id> --execute"`.
run-cluster: up-cluster
	$(call RUN_COMPOSE,--profile $(CLUSTER_PROFILE) exec -T $(CLUSTER_SERVICE) python -m janus.main --environment $(CLUSTER_PROFILE) --with-spark $(RUN_ARGS))

# The atomicity and engine-neutrality suites, re-run against the Postgres catalog and the
# object-store warehouse instead of the local SQLite one (AC-5 on the stack a deployment
# would actually use).
test-cluster: up-cluster
	$(call RUN_COMPOSE,--profile $(CLUSTER_PROFILE) exec -T -e JANUS_CLUSTER_SUITE=1 $(CLUSTER_SERVICE) python -m pytest -ra tests/integration/catalog_commits)

up-cluster-rest: check-compose seed-ivy seed-cluster-jars cluster-secrets
	$(call RUN_COMPOSE,--profile $(CLUSTER_REST_PROFILE) up -d || { printf '\n=== the stack did not come up; the catalog service log follows ===\n' >&2; $$compose_cmd $$compose_files --profile $(CLUSTER_REST_PROFILE) logs --no-color nessie >&2; $$compose_cmd $$compose_files --profile $(CLUSTER_REST_PROFILE) ps -a >&2; exit 1; })

down-cluster-rest: check-compose
	$(call RUN_COMPOSE,--profile $(CLUSTER_REST_PROFILE) down)

status-cluster-rest: check-compose
	$(call RUN_COMPOSE,--profile $(CLUSTER_REST_PROFILE) ps -a)

logs-cluster-rest: check-compose
	$(call RUN_COMPOSE,--profile $(CLUSTER_REST_PROFILE) logs --no-color nessie postgres minio minio-init)

shell-cluster-rest: up-cluster-rest
	$(call RUN_COMPOSE,--profile $(CLUSTER_REST_PROFILE) exec $(CLUSTER_REST_SERVICE) sh)

run-cluster-rest: up-cluster-rest
	$(call RUN_COMPOSE,--profile $(CLUSTER_REST_PROFILE) exec -T $(CLUSTER_REST_SERVICE) python -m janus.main --environment $(CLUSTER_PROFILE) --with-spark $(RUN_ARGS))

test-cluster-rest: up-cluster-rest
	$(call RUN_COMPOSE,--profile $(CLUSTER_REST_PROFILE) exec -T -e JANUS_CLUSTER_REST_SUITE=1 $(CLUSTER_REST_SERVICE) python -m pytest -ra tests/integration/catalog_commits)

docker-build: bootstrap

docker-run: run-local

clean:
	rm -rf .pytest_cache .ruff_cache
