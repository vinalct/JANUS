# Receita Federal CNPJ Integration

This note records the CNPJ bulk-file sources added for Receita Federal.

The source group is declarative. There is no Receita-specific hook and no custom Python branch by source id. Each entity uses the reusable file strategy, the shared link resolver chain, explicit schema contracts, and standard Spark read options.

## Source Set

The manifests live under `conf/sources/receita_federal/cnpj/` and currently cover:

- `empresas`
- `estabelecimentos`
- `socios`
- `simples`
- `cnaes`
- `motivos`
- `municipios`
- `naturezas`
- `paises`
- `qualificacoes`

Each source is disabled by default and follows the same shape:

- `source_type: file`
- `strategy: file`
- `strategy_variant: static_file`
- `access.link_resolver: nextcloud_webdav`
- `access.format: binary`
- `outputs.raw.format: binary`
- `outputs.bronze.format: iceberg`

The sources point at the public Receita Federal share and use `access.remote_file_pattern` to select only the ZIP archives for the configured entity. That keeps the selection in YAML instead of in a source hook.

## Schema And Read Options

The upstream files are headerless, semicolon-delimited CSVs encoded as `ISO-8859-1`. The source contracts declare that in `spark.read_options`:

```yaml
spark:
  input_format: csv
  read_options:
    header: "false"
    sep: ";"
    encoding: "ISO-8859-1"
    inferSchema: "false"
```

Column names come from active contracts under `conf/contracts/receita_federal/cnpj_<entity>.yaml`. During execution, JANUS generates the Spark read schema from each contract, so bronze tables get stable business column names instead of Spark's default positional names for headerless CSV.

## Output Layout

Raw archives and extracted members land under:

```text
data/raw/receita_federal/cnpj/<entity>
```

Bronze outputs are Iceberg tables in:

```text
bronze__receita_federal.cnpj_<entity>
```

Metadata, validation reports, checkpoints, lineage, and dead-letter state land under:

```text
data/metadata/receita_federal/cnpj/<entity>
```

## Operational Notes

These sources exercise the newer file-source behavior:

- Nextcloud public-share discovery through WebDAV;
- remote candidate filtering through `access.remote_file_pattern`;
- ZIP preservation in raw storage;
- archive-member filtering before Spark handoff;
- explicit schema use during Spark reads;
- dead-letter tolerance for failed file candidates.

The result is a reusable pattern for public bulk-file sources that publish several archive families from the same landing page.
