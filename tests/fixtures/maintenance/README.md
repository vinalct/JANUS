The zone is built by `tests.support.maintenance_zone.build_maintenance_zone(root, now)`;
no wall clock or checked-in dated metadata tree is used. `policy.json` is the explicit
policy used by all zone tests. The builder declares expected candidates independently
of the future collector/planner, and verifies all six AC-3 protected cases exist.

Two sources each have 25 runs and four artifact families, orphan lineage, current
state, five event days, and progress (one sanitized raw prefix, one legacy record).
The shared root has old/recent/null-timestamp pipeline summaries. Invalid JSON,
unparseable timestamps and atomic temporary leftovers remain protected.
