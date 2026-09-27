# Pinned Open Data Contract Standard schema

JANUS pins the Open Data Contract Standard (ODCS) JSON Schema at **v3.2.0**. This was the latest
released v3 version when it was pinned on 2026-09-22.

- Source: <https://github.com/bitol-io/open-data-contract-standard/blob/v3.2.0/schema/odcs-json-schema-v3.2.0.json>
- Release: <https://github.com/bitol-io/open-data-contract-standard/releases/tag/v3.2.0>
- Upstream commit: `f0bdad95346905d500be5ef4b2c2d9b1d95223b7`
- SHA-256: `edb41f33ec46e84780e99872ab2bd67f074959d2bf3e9c9fc54e61f8982b0d93`
- License: Apache-2.0

Verify the vendored bytes from the repository root:

```sh
sha256sum -c docs/schemas/odcs/*.sha256
```

For this reason, upgrading is a deliberate change with a diff. An upgrade must replace the schema
and sidecar, update the version expected by the loader, and revalidate every checked-in contract.

## JANUS subset

The runtime loader reads only the subset below. The pinned schema is authoritative for the ODCS
container shapes: `schema`, `customProperties`, and JANUS's chosen `team` representation are arrays.
ODCS v3.2.0 also supports the newer team object, but retains the v3.0.x team-member array as a
deprecated, valid representation; JANUS consistently uses the array form.

| ODCS key | JANUS use | Required |
|---|---|---|
| `apiVersion` | Must equal `v3.2.0`, the pinned version string. | yes |
| `kind` | Must equal `DataContract`. | yes |
| `id` | Stable contract id; convention: `<domain>.<bronze table>`. | yes |
| `name` | Human-readable contract name. | yes |
| `version` | Contract version as semver `MAJOR.MINOR.PATCH`. | yes |
| `status` | One of `draft`, `active`, or `deprecated`; this is a JANUS restriction of the ODCS free string. | yes |
| `domain` | Must equal the referencing source entry's `domain`. | yes |
| `description.purpose` | One paragraph used by the OpenLineage documentation facet. | yes |
| `tags` | List of strings. | no |
| `team` | List of `{username, role}` members with at least one `owner` role. | yes |
| `schema` | Exactly one element whose `name` is the Bronze table name, whose `physicalType` is `table`, and which contains `properties`. | yes |
| `schema[0].properties[]` | Bronze field metadata: `name`, `businessName`, `description`, derived `logicalType`, JANUS `physicalType`, `required`, `unique`, `primaryKey`, `classification`, and `customProperties` entries for `sourceField` and `sourceFormat`. A `struct` may contain nested `properties`; an `array` may contain `items`. | `name` and `physicalType`; other values have JANUS defaults |
| `customProperties` | Pairs for `janus.compatibility` (`additive`, `backward`, or `frozen`), `janus.enforcement` (`strict` or `lenient`), and optionally `janus.draftedFrom` (free text on drafts only). | compatibility and enforcement |

Every `customProperties` value uses ODCS's array-of-pairs form:

```yaml
customProperties:
  - property: janus.compatibility
    value: additive
```

ODCS keys outside this subset, including `servers`, `slaProperties`, `price`, `quality`, `roles`, and
`support`, are allowed when the pinned schema allows them and are ignored by the JANUS loader. The
loader does not reject an ODCS key merely because JANUS does not consume it. It does reject unknown
custom-property names with the `janus.` prefix so misspelled JANUS behavior cannot be silently lost.
