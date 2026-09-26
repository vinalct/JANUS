# INEP Microdata Integration

This note records what was delivered for the first INEP source and the extra operational work that was needed after the first real container run.

The source itself is simple on paper: INEP publishes the Censo Escolar microdata as a public ZIP file. In practice, it was useful because it exercised a different part of JANUS than the API sources. The unit of extraction is not a JSON page or a catalog record. It is a bulk archive that must be preserved, unpacked, filtered, and handed to Spark in a deterministic way.

## What was delivered

The new source is defined in `conf/sources/inep/inep.yaml` as `inep_censo_escolar_microdados`.

It is intentionally modeled as:

- `strategy: file`
- `strategy_variant: archive_package`
- `extraction.mode: snapshot`
- `checkpoint_strategy: none`

There is no source hook for INEP. That was a deliberate choice. The source fits the reusable file strategy well enough, so the implementation stays in configuration, schema, fixtures, and tests instead of adding another source-specific Python path.

The source points to:

```text
https://download.inep.gov.br/dados_abertos/microdados_censo_escolar_2024.zip
```

The configured package member is:

```text
*/dados/microdados_ed_basica_2024.csv
```

That keeps the original ZIP in raw storage while selecting only the Censo Escolar CSV for the normalization handoff.

The output layout follows the same JANUS zones used by the other sources:

- raw archive and extracted files: `data/raw/inep/censo_escolar_microdados`
- bronze Iceberg output: `bronze_inep.censo_escolar_microdados`
- run metadata, lineage, and related records: `data/metadata/inep/censo_escolar_microdados`

The source is disabled by default. Running it requires `--include-disabled`, which is the safer default for a bulk download source during development.

## Schema and quality contract

The explicit schema lives at:

```text
conf/contracts/educacao/censo_escolar_microdados.yaml
```

The first schema is intentionally narrow. It covers the fields needed to prove the pipeline contract and quality checks without trying to model the entire INEP release in one pass.

The required fields currently are:

- `NU_ANO_CENSO`
- `CO_ENTIDADE`
- `TP_DEPENDENCIA`

This gives the integration a stable quality gate while leaving room to expand the schema after the raw-to-bronze path is already proven.

## Tests added for the source

The integration test builds a small ZIP fixture during the test run instead of storing a large upstream archive in the repository.

The test coverage checks that:

- the registry loads the INEP source contract correctly;
- the source uses the generic `FileStrategy`;
- the source has no source hook;
- the archive is persisted under a deterministic raw download path;
- the selected CSV member is extracted under a deterministic raw extracted path;
- the normalization handoff receives the CSV, not the ZIP;
- bronze Iceberg and metadata can be materialized when Spark is available;
- reruns reuse the same raw paths.

The fixture lives under:

```text
tests/fixtures/inep/
```

The integration tests live under:

```text
tests/integration/inep/
```

## Archive behavior

The file strategy preserves both sides of the package:

- the original upstream ZIP, written under `downloads/<version>/`;
- the extracted CSV member, written under `extracted/<version>/`.

For the 2024 file, the resolved version is `2024`, taken from the package name. That gives repeatable paths and makes local reruns easier to compare.

The selected member path contains the archive root directory and the internal `dados` directory. That is expected. JANUS keeps the upstream archive layout visible in raw storage instead of flattening it and risking name collisions later.

## Runtime notes

The INEP file is a ZIP download, so the access format is `binary`. The Spark handoff reads the extracted CSV.

The source contract reads the CSV using:

```text
header=true
sep=;
inferSchema=true
```

That reflects the shape of the INEP CSV. The source contract declares both `spark.input_format: csv` and the CSV reader options under `spark.read_options`, so the same parser settings drive the framework execution path and the raw-to-bronze reload path.

## Certificate issue found during the first real run

The first container execution failed before any data parsing happened. The download request failed with:

```text
SSL: CERTIFICATE_VERIFY_FAILED
unable to get local issuer certificate
```

This was not caused by the JANUS source config, auth, Docker networking, VPN, or firewall.

The live INEP download host was checked with OpenSSL, curl, and Python. The host presented only the leaf certificate:

```text
subject: CN=*.inep.gov.br
issuer:  CN=RNP ICPEdu GR46 OV TLS CA 2025
```

The server did not send the intermediate certificate needed to build the chain to the already trusted GlobalSign root. Many browsers can recover from this by fetching the missing intermediate. Python, curl, and OpenSSL do not reliably do that in batch jobs, which is why JANUS failed while other API sources still worked.

The missing intermediate is:

```text
subject: C=BR, O=REDE NACIONAL DE ENSINO E PESQUISA - RNP, CN=RNP ICPEdu GR46 OV TLS CA 2025
issuer:  C=BE, O=GlobalSign nv-sa, CN=GlobalSign Root R46
sha256: E1:07:47:D4:DA:7B:AB:09:CB:A9:95:2F:01:9D:35:34:CB:9F:BA:07:0B:F1:3D:87:91:B1:69:9C:D2:FF:59:DD
valid until: 2030-11-19
```

The certificate was added to:

```text
conf/certs/rnpicpedugr46ovtlsca2025.pem
```


## TLS adaptation

The fix keeps TLS verification enabled.

Two runtime adaptations were made:

- the Docker image installs the intermediate into the container trust store with `update-ca-certificates`;
- `docker-compose.yml` sets `JANUS_CA_BUNDLE=/workspace/conf/certs/rnpicpedugr46ovtlsca2025.pem`, so JANUS explicitly loads the intermediate when building its Python SSL context.

The HTTP transport also supports explicit CA bundle candidates. That gives the project a controlled way to add trusted CA material when a public source has a broken or incomplete chain.

This is intentionally different from unsafe workarounds such as:

- disabling SSL verification;
- using an unverified Python SSL context;
- switching the source URL to plain HTTP;
- hiding certificate handling inside source-specific code.

The INEP endpoint also responds over HTTP, but the source remains on HTTPS. Keeping HTTPS matters because the raw zone should preserve what was fetched from a verified upstream endpoint.

## How to run after the adaptation

After changing the certificate or Docker trust setup, rebuild the image instead of only restarting the container:

```sh
make down
docker compose -f docker/docker-compose.yml build --no-cache janus
make up
```


## Maintenance notes

This certificate workaround should be revisited if INEP fixes the server chain or rotates to a different issuer.

The current intermediate expires on 2030-11-19. If downloads start failing again with the same certificate error, check the live chain before changing JANUS code. The right fix is to trust the missing public intermediate or remove this workaround if the server begins sending a complete chain.

Do not replace this with disabled verification. That would make the pipeline easier to run, but it would weaken the provenance guarantee of the raw data.
