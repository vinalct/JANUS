COMPOSE_FILE := docker/docker-compose.yml
PODMAN_COMPOSE_FILE := docker/docker-compose.podman.yml
SERVICE := janus
ENVIRONMENT ?= local
JANUS_UID := $(shell id -u)
JANUS_GID := $(shell id -g)
JANUS_PROJECT_ROOT := $(CURDIR)

# Iceberg runtime jar is vendored under deps/ (data/metadata/** is ignored) and seeded into the
# path the Spark session resolves from, so Iceberg never resolves over the network.
IVY_JAR_NAME := org.apache.iceberg_iceberg-spark-runtime-4.0_2.13-1.10.1.jar
IVY_JAR_SRC := deps/$(IVY_JAR_NAME)
IVY_JAR_DEST_DIR := data/metadata/ivy/jars

DETECT_COMPOSE = if podman compose version >/dev/null 2>&1; then echo 'podman compose'; elif command -v podman-compose >/dev/null 2>&1; then echo podman-compose; elif docker compose version >/dev/null 2>&1; then echo 'docker compose'; elif command -v docker-compose >/dev/null 2>&1; then echo docker-compose; else exit 1; fi

define RUN_COMPOSE
	@compose_cmd="$$( $(DETECT_COMPOSE) )" || { \
		echo "No compose-capable container engine found. Install Docker Compose, docker-compose, podman compose, or podman-compose." >&2; \
		exit 1; \
	}; \
	compose_files="$$(case "$$compose_cmd" in podman* ) printf '%s' '-f $(COMPOSE_FILE) -f $(PODMAN_COMPOSE_FILE)' ;; * ) printf '%s' '-f $(COMPOSE_FILE)' ;; esac)"; \
	container_user="$$(case "$$compose_cmd" in podman* ) printf '%s:%s' '$(JANUS_UID)' '$(JANUS_GID)' ;; * ) printf '%s:%s' '$(JANUS_UID)' '$(JANUS_GID)' ;; esac)"; \
	JANUS_CONTAINER_USER=$$container_user JANUS_UID=$(JANUS_UID) JANUS_GID=$(JANUS_GID) JANUS_PROJECT_ROOT=$(JANUS_PROJECT_ROOT) $$compose_cmd $$compose_files $(1)
endef

.PHONY: bootstrap check-compose up ensure-up seed-ivy down status logs shell pyspark-local lint typecheck test ci run-local run-local-config docker-build docker-run clean

seed-ivy:
	@if [ ! -f "$(IVY_JAR_DEST_DIR)/$(IVY_JAR_NAME)" ]; then \
		echo "Seeding Iceberg runtime jar from $(IVY_JAR_SRC)"; \
		mkdir -p "$(IVY_JAR_DEST_DIR)"; \
		cp "$(IVY_JAR_SRC)" "$(IVY_JAR_DEST_DIR)/$(IVY_JAR_NAME)"; \
	fi

check-compose:
	@compose_cmd="$$( $(DETECT_COMPOSE) )" || { \
		echo "No compose-capable container engine found. Install Docker Compose, docker-compose, podman compose, or podman-compose." >&2; \
		exit 1; \
	}; \
	printf 'Using %s\n' "$$compose_cmd"; \
	case "$$compose_cmd" in podman* ) printf 'Using Podman keep-id user namespace for writable bind mounts\n' ;; esac

bootstrap: check-compose
	$(call RUN_COMPOSE,build $(SERVICE))

# Explicit fresh restart — stops and recreates the container.
up: check-compose seed-ivy
	$(call RUN_COMPOSE,up -d --force-recreate $(SERVICE))

ensure-up: check-compose seed-ivy
	$(call RUN_COMPOSE,up -d $(SERVICE))

down: check-compose
	$(call RUN_COMPOSE,down)

status: check-compose
	$(call RUN_COMPOSE,ps)

logs: check-compose
	$(call RUN_COMPOSE,logs $(SERVICE))

shell: ensure-up
	$(call RUN_COMPOSE,exec $(SERVICE) sh)

pyspark-local: ensure-up
	$(call RUN_COMPOSE,exec $(SERVICE) sh -lc '\
	ivy_dir="$${JANUS_SPARK_IVY_DIR:-data/metadata/ivy}"; \
	if [ "$$ivy_dir" = "$${ivy_dir#/}" ]; then ivy_dir="/workspace/$$ivy_dir"; fi; \
	iceberg_warehouse="$${JANUS_ICEBERG_WAREHOUSE_DIR:-data/bronze/iceberg}"; \
	if [ "$$iceberg_warehouse" = "$${iceberg_warehouse#/}" ]; then iceberg_warehouse="/workspace/$$iceberg_warehouse"; fi; \
	spark_warehouse="$${JANUS_SPARK_WAREHOUSE_DIR:-data/metadata/spark-warehouse}"; \
	if [ "$$spark_warehouse" = "$${spark_warehouse#/}" ]; then spark_warehouse="/workspace/$$spark_warehouse"; fi; \
	pyspark \
		--packages "$${JANUS_ICEBERG_RUNTIME_PACKAGE:-org.apache.iceberg:iceberg-spark-runtime-4.0_2.13:1.10.1}" \
		--conf spark.jars.ivy="$$ivy_dir" \
		--conf spark.sql.extensions=org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions \
		--conf spark.sql.defaultCatalog="$${JANUS_ICEBERG_CATALOG_NAME:-janus}" \
		--conf spark.sql.catalog.$${JANUS_ICEBERG_CATALOG_NAME:-janus}=org.apache.iceberg.spark.SparkCatalog \
--conf spark.sql.catalog.$${JANUS_ICEBERG_CATALOG_NAME:-janus}.type=hadoop \
--conf spark.sql.catalog.$${JANUS_ICEBERG_CATALOG_NAME:-janus}.warehouse="$$iceberg_warehouse" \
--conf spark.sql.catalog.$${JANUS_ICEBERG_CATALOG_NAME:-janus}.default-namespace="$${JANUS_ICEBERG_DEFAULT_NAMESPACE:-bronze}" \
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

# Reproduce CI locally: same lint + type check + full suite the container CI job runs,
# in the container so the Spark/Iceberg path is exercised. Keep in lockstep with .github/workflows/ci.yml.
ci: ensure-up
	$(call RUN_COMPOSE,exec -T $(SERVICE) python -m ruff check src tests)
	$(call RUN_COMPOSE,exec -T $(SERVICE) python -m mypy)
	$(call RUN_COMPOSE,exec -T $(SERVICE) python -m pytest -ra --cov=janus --cov-report=term-missing)

run-local: ensure-up
	$(call RUN_COMPOSE,exec -T $(SERVICE) python -m janus.main --environment $(ENVIRONMENT) --with-spark)

run-local-config: ensure-up
	$(call RUN_COMPOSE,exec -T $(SERVICE) python -m janus.main --environment $(ENVIRONMENT))

docker-build: bootstrap

docker-run: run-local

clean:
	rm -rf .pytest_cache .ruff_cache
