# OO Sovereign Handoff Format

## Purpose

This document defines the first exchange format produced by `oo-host` for future consumption by the sovereign runtime.

The objective is modest and concrete:

- provide a compact, versioned, machine-readable summary,
- preserve organism continuity markers,
- carry policy posture,
- expose top active goals,
- expose recent causal events,
- remain simple enough to parse later from `llm-baremetal`.

---

## 1. File intent

Suggested filename:

- `sovereign_export.json`

Meaning:

- a host-side handoff summary for sovereign import or inspection
- not a full state image
- not an authoritative replacement for sovereign runtime state
- not a raw journal dump

---

## 2. Design rules

The format must be:

- JSON
- versioned
- appendage-safe (new fields may be added later)
- compact enough for constrained readers
- causal enough to explain recent state

The format must not:

- require the sovereign runtime to understand every host detail
- contain unbounded journal history
- contain host-specific opaque handles

---

## 3. Top-level fields

## Required

- `schema_version`
- `export_kind`
- `generated_at_epoch_s`
- `organism_id`
- `genesis_id`
- `runtime_habitat`
- `runtime_instance_id`
- `continuity_epoch`
- `boot_or_start_count`
- `mode`
- `policy`
- `active_goal_count`
- `top_goals`
- `recent_events`

## Optional

- `last_recovery_reason`

---

## 4. Expected values

### `schema_version`

Current value:

- `1`

### `export_kind`

Current value:

- `oo_sovereign_handoff`

### `runtime_habitat`

Examples:

- `host_windows`
- `host_macos`
- `host_linux`

### `mode`

Expected values:

- `normal`
- `degraded`
- `safe`

---

## 5. Policy object

Expected fields:

- `safe_first`
- `deny_by_default`
- `llm_advisory_only`
- `enforcement`

Expected `enforcement` values:

- `off`
- `observe`
- `enforce`

---

## 6. Goal object

Each entry in `top_goals` should expose:

- `goal_id`
- `title`
- `status`
- `priority`
- `safety_class`

Rule:

- only a bounded number of top goals should be exported
- completed/aborted goals should normally be excluded

---

## 7. Recent event object

Each entry in `recent_events` should expose:

- `ts_epoch_s`
- `kind`
- `severity`
- `summary`
- `reason`
- `action`
- `result`
- `continuity_epoch`

Rule:

- the event list is a compact causal tail, not the full journal
- the exported tail should remain bounded

---

## 8. Current semantic intent

The sovereign runtime should eventually be able to use this handoff for:

- inspection
- continuity checks
- mode-awareness
- high-priority goal awareness
- policy posture awareness
- recent recovery/policy event awareness

It should **not** blindly import host conclusions as sovereign truth.

The sovereign side remains authoritative for its own survival and recovery logic.

---

## 9. Example shape

```json
{
  "schema_version": 1,
  "export_kind": "oo_sovereign_handoff",
  "generated_at_epoch_s": 1773442521,
  "organism_id": "...",
  "genesis_id": "...",
  "runtime_habitat": "host_windows",
  "runtime_instance_id": "...",
  "continuity_epoch": 0,
  "boot_or_start_count": 12,
  "mode": "normal",
  "last_recovery_reason": null,
  "policy": {
    "safe_first": true,
    "deny_by_default": true,
    "llm_advisory_only": true,
    "enforcement": "observe"
  },
  "active_goal_count": 2,
  "top_goals": [
    {
      "goal_id": "...",
      "title": "bootstrap organism memory",
      "status": "pending",
      "priority": 0,
      "safety_class": "normal"
    }
  ],
  "recent_events": [
    {
      "ts_epoch_s": 1773442519,
      "kind": "goal_complete",
      "severity": "notice",
      "summary": "goal done: bootstrap organism memory",
      "reason": null,
      "action": "goal_done",
      "result": "ok",
      "continuity_epoch": 0
    }
  ]
}
```

---

## 10. Reader expectations for future llm-baremetal support

A future reader in `llm-baremetal` should:

- reject unknown `export_kind`
- reject unsupported `schema_version`
- fail closed on malformed JSON
- treat missing optional fields as non-fatal
- never confuse the handoff file with a model or policy artifact
- use the file as advisory/importable summary only

### Current import policy for `/oo_handoff_apply`

The first active sovereign import remains intentionally narrow.

Imported fields:

- `mode`
- `policy.enforcement`
- `continuity_epoch` (recorded locally as the latest observed host continuity marker)
- `last_recovery_reason` (recorded locally when present)

Safety rules:

- sovereign mode may only move toward a **safer** local posture
  - `normal -> degraded -> safe`
  - a host request to become less strict is ignored
- sovereign policy enforcement may only move toward a **stricter** local posture
  - `off -> observe -> enforce`
  - a host request to weaken local enforcement is ignored
- malformed or unsupported values are rejected
- organism metadata is currently inspected and journaled, not trusted as an authority switch
- continuity markers are currently persisted as a local handoff receipt, not as authoritative sovereign state replacement

Operational note:

- the persisted receipt can be inspected from sovereign runtime with `/oo_handoff_receipt`
- the sovereign/runtime comparison view is available with `/oo_continuity_status`

This means the host can warn or tighten, but cannot silently relax sovereign survival posture.

---

## 11. Current decision

The first host → sovereign exchange format is now a:

- versioned,
- bounded,
- causal,
- policy-aware JSON handoff.
