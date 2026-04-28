# REPL Commands (llm-baremetal)

This file is a “complete” cheat-sheet of the commands available in the UEFI REPL.

## Keyboard shortcuts

- **Enter**: submit the line (or the full prompt)
- **Backspace**: delete one character
- **Up / Down arrows**: command history (single-line only)
- **Tab**: auto-complete `/...` commands (press repeatedly to cycle matches, single-line only)

## Multi-line input

- End a line with `\` to continue on the next line.
- Type `;;` on a line by itself to **submit** the multi-line block.
- If you want a literal trailing backslash, end the line with `\\`.

## Sampling / generation

- `/temp <val>`: temperature (0.0=greedy, 1.0=creative)
- `/min_p <val>`: min_p (0.0–1.0, 0=off)
- `/top_p <val>`: nucleus sampling (0.0–1.0)
- `/top_k <int>`: top-k (0=off)
- `/norepeat <n>`: no-repeat ngram (0=off)
- `/repeat <val>`: repeat penalty (1.0=none)
- `/max_tokens <n>`: max generated tokens (1–256)
- `/seed <n>`: RNG seed
- `/stats <0|1>`: print generation stats
- `/stop_you <0|1>`: stop on the `\nYou:` pattern
- `/stop_nl <0|1>`: stop on double newline

## Info / debug

- `/version`: version + build + features
- `/ctx`: show model + sampling + budgets
- `/cfg`: show effective repl.cfg settings
- `/diag`: display system diagnostics (GOP/RAM/CPU/models)
- `/model`: loaded model info
- `/model_info [file]`: show file header/metadata (supports `.bin` and `.gguf`, and resolves root/models plus FAT 8.3 aliases)
- Note: GGUF inference supports **F16/F32/Q4_0/Q4_1/Q5_0/Q5_1/Q8_0**.
  - Note: `Q4_K_*` / `Q5_K_*` (K-quants) are not supported yet.
  - Default behavior: tensors are dequantized to float32 at load.
  - `gguf_q8_blob=1`: keep Q8_0 matrices quantized in RAM (faster load + lower RAM).
  - `q8_act_quant` (only relevant for Q8_0 blob + AVX2):
    - `0`: off (highest fidelity)
    - `1`: on for all Q8 matmuls (fastest, most approximation)
    - `2`: FFN-only (w1/w3/w2 use i8 dot; attention projections stay float for better quality/perf tradeoff)
- `/models [dir]`: list available model files (`.bin` / `.gguf`) in root and `models\\`
  - If long filenames fail to open on your firmware, use an 8.3-compatible name (or the `NAME~1.EXT` alias) in `repl.cfg: model=...`.
- `/cpu`: SIMD status
- `/attn [auto|sse2|avx2]`: force the attention SIMD path
- `/zones`: dump allocator zones + sentinel
- `/budget [p] [d]`: budgets in cycles (prefill, decode)
- `/test_failsafe [prefill|decode|both] [cycles]`: one-shot strict_budget trip
- `/commands [filter]`: list commands (filter is case-insensitive substring; if it starts with `/` it's a prefix)
  - examples: `/commands dump` (matches `/save_dump`), `/commands /oo_`
- `/help [filter]`: help (same filtering rules)
  - examples: `/help save`, `/help /oo_`

## Bench / perf

- `/blas_bench`: float32 matmul benchmark (scalar vs SIMD)
- `/q8_bench [n] [d] [reps]`: synthetic Q8_0 matmul benchmark (scalar vs AVX2)
  - prints `AVX2(i8)` when `q8_act_quant!=0`
- `/q8_matvec [wq|wk|wv|wo|w1|w2|w3|cls] [layer] [reps]`: real model Q8_0 matvec benchmark (requires Q8 blob weights)
  - prints `AVX2(i8)` when the selected matrix is using the i8 path (all matrices for `q8_act_quant=1`, FFN-only for `q8_act_quant=2`)

## Logs / dumps

- `/log [n]`: print the last n log entries
- `/save_log [n]`: write the last n log entries to `llmk-log.txt`
- `/save_dump`: write ctx+zones+sentinel+log to `llmk-dump.txt`
- `/diag_status`: show Diagnostion status
- `/diag_report [file]`: write a full diagnostic bundle with system info, CPU/GOP/RAM, and model inventory (default `llmk-diag.txt`)
- `/mem_status`: show Memorion status
- `/mem_snap_info [file]`: print snapshot header (default `llmk-snap.bin`)
- `/mem_snap_check [file]`: check snapshot compatibility vs current model
- `/mem_manifest [snap] [out]`: write manifest (default out `llmk-manifest.txt`)

## Orchestrion (workflow runner)

- `/orch_on`: enable Orchestrion (observe mode)
- `/orch_off`: disable Orchestrion
- `/orch_enforce [0|1|2]`: set mode (0=off, 1=observe, 2=enforce)
- `/orch_status`: show pipeline state + counters
- `/orch_clear`: clear pipeline
- `/orch_add <step> [; step2 ...]`: add step(s) to pipeline
- `/orch_start [loops]`: start pipeline (default 1 loop)
- `/orch_pause`: pause pipeline
- `/orch_resume`: resume pipeline
- `/orch_stop`: stop pipeline

## Calibrion (auto-tuning sampling)

- `/calib_on`: enable Calibrion (observe mode)
- `/calib_off`: disable Calibrion
- `/calib_enforce [0|1|2]`: set mode (0=off, 1=observe, 2=enforce)
- `/calib_strategy <none|entropy|length|quality|hybrid>`: set strategy
- `/calib_status`: show stats + recommendation
- `/calib_reset`: reset stats
- `/calib_apply`: apply recommendation to temp/top_k/top_p

## Compatibilion (platform detection)

- `/compat_on`: enable Compatibilion
- `/compat_off`: disable Compatibilion
- `/compat_status`: show CPU/platform capabilities + recommendations
- `/compat_probe`: re-probe CPU features

## GOP / rendering

- `/gop`: GOP framebuffer info
- `/tui_on`: enable the GOP TUI overlay (status panel)
- `/tui_off`: disable the GOP TUI overlay
- `/tui_toggle`: toggle the GOP TUI overlay
- `/tui_redraw`: force a redraw of the overlay
- `/tui_mode <status|log|split|files>`: set the GOP UI mode
- `/tui_log_on`: show the transcript log UI (same as `/tui_mode log`)
- `/tui_log_off`: return to status-only UI (same as `/tui_mode status`)
- `/tui_log_clear`: clear the transcript ring buffer
- `/tui_log_up [n]`: scroll transcript up (older lines)
- `/tui_log_down [n]`: scroll transcript down (newer lines)
- `/tui_log_dump [file]`: dump transcript to a text file (default `llmk-transcript.txt`)
- `/render <dsl>`: render simple shapes via DSL
- `/save_img [f]`: save GOP framebuffer as PPM (default `llmk-img.ppm`)
- `/draw <text>`: ask the model for DSL then execute `/render` (GOP required)

### GOP file browser

Minimal on-screen file browser rendered via GOP (works on the same FAT image / USB).

- `/fb` or `/fb_on`: enable the file browser pane
- `/fb_off`: disable the file browser pane
- `/fb_refresh`: refresh the directory listing
- `/fb_cd <dir>`: change directory
- `/fb_up`: go to parent directory
- `/fb_sel <n>`: select entry index `n`
- `/fb_open`: open selection (directories: enter; files: preview)

DSL quick ref:

- `clear R G B; rect X Y W H R G B; pixel X Y R G B`

## LLM-OO (organism-oriented)

- `/oo_new <goal>`: create an entity (long-lived intention)
- `/oo_list`: list entities
- `/oo_show <id>`: show an entity (goal/status/digest/notes tail)
- `/oo_kill <id>`: delete an entity
- `/oo_note <id> <text>`: append a note

Agenda:

- `/oo_plan <id> [prio] <action(s)>`: add actions (separator `;`, prio like `+2`)
- `/oo_agenda <id>`: show agenda
- `/oo_next <id>`: pick next action (marks “doing”)
- `/oo_done <id> <k>`: mark action #k done
- `/oo_prio <id> <k> <p>`: set priority for action #k
- `/oo_edit <id> <k> <text>`: edit action #k text

Execution:

- `/oo_step <id>`: advance one entity by one step
- `/oo_run [n]`: run n cooperative steps
- `/oo_digest <id>`: update digest + compress notes

Persistence:

- `/oo_save [f]`: save (default `oo-state.bin`)
- `/oo_load [f]`: load (default `oo-state.bin`)
- `/oo_reboot_probe`: arm a continuity probe, trigger a reboot, then verify on the next boot that `boot_count` advanced and local/recovery state stayed aligned
- Note: a `*.bak` backup is created best-effort before overwrite.

Think/auto:

- `/oo_think <id> <prompt>`: ask the model, store the answer in notes
- `/oo_exec <id> [n] [--plan] [hint]`: run agenda items for n cycles; stops when agenda is empty unless `--plan` (stop: `q` or Esc between cycles)
- `/oo_exec_stop`: stop exec mode
- `/oo_auto <id> [n] [prompt]`: n cycles think->store->step (stop: `q` or Esc between cycles)
- `/oo_auto_stop`: stop auto mode

## Operating Organism (OO) — Homeostasis

These commands allow the kernel to monitor and adapt its own state in response to resource pressure:

- `/oo_status`: show OO config, persistence artifacts, and the latest consult summary
- `/oo_log [n]`: tail `OOCONSULT.LOG` (latest consults, decisions, and persisted dynamics)
- `/oo_outcome [n]`: tail `OOOUTCOME.LOG`, pending next-boot checks, and confirmed adaptation outcomes
- `/oo_explain`: explain the latest consult decision in short form
- `/oo_explain verbose`: show the latest consult with confidence, plan, boot/trend/saturation dynamics, and operator summary
- `/oo_explain boot`: focus on latest confirmed boot comparison plus recent confirmed outcome history
- `/oo_consult_mock <text>`: run the consult policy with a deterministic mock suggestion for testing
- `/oo_consult`: ask the embedded LLM for system adaptation suggestions (M5/M5.1/M5.2 features)
  - LLM receives system state (mode, RAM, ctx_len, boots, journal tail).
  - LLM suggests ONE brief action (M5) or 1-3 actions (M5.1 if `oo_multi_actions=1`).
  - Policy engine applies safety-first rules:
    - **SAFE mode**: only reductions allowed.
    - **DEGRADED/NORMAL**: increases blocked if RAM < 1GB.
    - **Reboot/model changes**: logged but not auto-applied (v0).
  - Multi-action (M5.1): detects and applies multiple compatible actions (ex: "reduce ctx AND seq").
    - Priority rules: stable>reboot>reduce (reduce blocks increase).
    - Emits batch summary: `OK: OO policy batch: N applied, M blocked`.
  - Auto-apply (M5.2): `oo_auto_apply=0|1|2` controls automatic application.
    - Mode 0: simulation only (log "would_apply_if_enabled").
    - Mode 1: conservative (auto-apply reductions only).
    - Mode 2: aggressive (auto-apply reductions + increases if safe).
    - Throttling: 1 auto-apply per boot to prevent adaptation spirals.
    - Markers: `OK: OO auto-apply: reduce_ctx (old=512 new=256 check=pass)`.
  - Deterministic markers:
    - `OK: OO LLM suggested: <text>` (serial)
    - `OK: OO policy decided: <action> (reason=<reason>)` (serial)
    - `oo event=consult decision=<action> reason=<reason>` (journal)
  - Config: `oo_llm_consult=0|1` (default: follows `oo_enable` value)

Recent consult builds also expose higher-level operator fields in `/oo_status`, `/oo_log`, and `/oo_explain verbose`:

- `last.consult.boot_relation` / `boot_bias`
- `last.consult.trend` / `trend_bias`
- `last.consult.saturation` / `saturation_bias`
- `last.consult.operator_summary`

This makes it easier to understand cases such as `positive_but_saturated`, where history still favors an action but the target is already at its min/max bound.

## Autorun

- `/autorun_stop`: stop the current autorun
- `/autorun [--print] [--shutdown|--no-shutdown] [f]`
  - `--print`: print runnable lines from the script without executing
  - `--shutdown`: UEFI shutdown when the script completes
  - `--no-shutdown`: do not shutdown when the script completes
  - `f`: file name (default: `autorun_file` from `repl.cfg`, else `llmk-autorun.txt`)

### Autorun config (repl.cfg)

- `autorun_autostart=1` to start autorun at boot (disabled by default)
- `autorun_file=llmk-autorun.txt`
- `autorun_shutdown_when_done=0`

## Reset / context

- `/reset`: reset budgets/log + untrip sentinel
- `/clear`: clear KV cache (reset conversation context)

## File shell

Work with files directly from the UEFI REPL (FAT image / USB):

- `/fs_ls [dir]`: list directory (default: root)
- `/fs_cat <file>`: print a file (best-effort text; truncated)
- `/fs_write <file> <text...>`: truncate/create and write text
- `/fs_append <file> <text...>`: append text (create if missing)
- `/fs_rm <file>`: delete file
- `/fs_cp <src> <dst>`: copy file (best-effort)
- `/fs_mv <src> <dst>`: move file (copy+delete best-effort)

## Snapshot (fast resume)

Save/load the KV cache so you can continue a conversation after reboot without rebuilding context.

- `/snap_save [file]`: save snapshot (default: `llmk-snap.bin`)
- `/snap_load [file]`: load snapshot (default: `llmk-snap.bin`)

Convenience:

- `/snap_autoload_on [file]`: write `repl.cfg` to enable snapshot auto-load at boot (optional `snap_file` override)
- `/snap_autoload_off`: write `repl.cfg` to disable snapshot auto-load at boot

### Snapshot auto-resume config (repl.cfg)

- `snap_autoload=1` to attempt loading a snapshot at boot (disabled by default)
- `snap_file=llmk-snap.bin` (optional override; default is `llmk-snap.bin`)

Notes:

- Snapshot files grow with `kv_pos` (only the used prefix is stored).
- Snapshots are model-config dependent (must match dim/layers/heads/seq_len).

## DjibMark

- `/djibmarks`: DjibMark trace
- `/djibperf`: performance analysis by phase

## Djibion (meta-engine of coherence)

Djibion is a lightweight policy/validation engine that can gate critical actions.

Currently, it can gate/transform:

- filesystem mutations (`/fs_write`, `/fs_append`, `/fs_rm`)
- filesystem copy/move (`/fs_cp`, `/fs_mv`) with destination prefix transform
- OO cycles (`/oo_exec`, `/oo_auto`) via `max_oo_cycles`
- OO persistence (`/oo_save`, `/oo_load`) via `allow_oo_persist`
- autorun (`/autorun`) with optional prefix enforcement
- snapshots (`/snap_save`, `/snap_load`, plus boot `snap_autoload=1`) with traversal protection and optional prefix transform
- repl.cfg writes (e.g. `/snap_autoload_on`, `/snap_autoload_off`) via `allow_cfg_write`

- `/djibion_on`: enable Djibion in observe mode (logs decisions, does not block)
- `/djibion_off`: disable Djibion
- `/djibion_enforce <0|1|2>`: set mode (0=off, 1=observe, 2=enforce)
- `/djibion_status`: show current laws + counters
- `/djibion_prefix <prefix>`: set allowed prefix for file actions (example: `\\test_dir\\`)
- `/djibion_allow_delete <0|1>`: allow deleting files
- `/djibion_max_write <bytes>`: set max bytes for `/fs_write` and `/fs_append`
- `/djibion_max_oo <n>`: set max cycles per `/oo_exec` or `/oo_auto`
- `/djibion_allow_autorun <0|1>`: allow `/autorun`
- `/djibion_allow_oo_persist <0|1>`: allow `/oo_save` and `/oo_load`

### Djibion config (repl.cfg)

You can enable Djibion governance at boot by setting keys in `repl.cfg`:

- `djibion_mode=0|1|2` (0=off, 1=observe, 2=enforce)
- `djibion_prefix=\\test_dir\\` (optional prefix restriction)
- `djibion_allow_write=0|1`
- `djibion_allow_delete=0|1`
- `djibion_max_write=<bytes>`
- `djibion_max_oo=<n>`
- `djibion_allow_autorun=0|1`
- `djibion_allow_oo_persist=0|1`
- `djibion_allow_snap_load=0|1`
- `djibion_allow_snap_save=0|1`
- `djibion_max_snap=<bytes>`
- `djibion_allow_cfg_write=0|1`

## Diopion (speed / burst engine)

Diopion is a complementary engine focused on “bursty exploration” (temporary sampling knob overrides).
It does not bypass Djibion safety gates; it only tweaks generation parameters.

- `/diopion_on`: enable Diopion (observe mode)
- `/diopion_off`: disable Diopion (also cancels any active burst)
- `/diopion_enforce <0|1|2>`: set mode (0=off, 1=observe, 2=enforce)
- `/diopion_profile <none|animal|vegetal|geom|bio>`: apply a preset profile (v0.1)
- `/diopion_burst [turns] [temp_milli] [top_k] [max_tokens]`: start/refresh a burst
- `/diopion_status`: show current mode/profile + burst defaults

### Diopion config (repl.cfg)

- `diopion_mode=0|1|2`
- `diopion_profile=none|animal|vegetal|geom|bio`
- `diopion_burst_turns=<n>` (1–16)
- `diopion_burst_max_tokens=<n>` (16–1024)
- `diopion_burst_topk=<n>` (1–200)
- `diopion_burst_temp_milli=<n>` (50–2000, e.g. 900 => 0.900)
