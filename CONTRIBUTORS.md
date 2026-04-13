# Contributors & Credits

## Core Architecture

### ItsMick (@ItsMick)
**O(1) Stateful Loop Engine** — Original author of the `feat/stateful-loop-engine` branch.
Conceived and implemented the fundamental architectural shift from O(n) re-tokenization
to true O(1) MambaCache recurrent steps, achieving 2.35x–3.17x latency improvements.
Files: `stateful_engine.py` (original), `validate_stateful.py`, `benchmark_llps.py`,
`docs/cache_api_findings.md`, `docs/llps_benchmark.md`, `CONTRIBUTION.md`

### Djiby Diop (@Djiby-diop)
**llm-baremetal / OO-SomaMind** — Author of the UEFI bare-metal bootloader and
the OOSI v3 binary format, the BATTERFYL training specification, and the Operating
Organism architecture (D+ Policy Engine, engine names, DNA hash system, phase A-Z,
REPL commands `/ssm_infer`, `/oo_status`, `/zones`).
Repository: https://github.com/Djiby-diop/llm-baremetal

### GitHub Copilot AI
**Code Review** — Identified 3 critical API bugs in the original PR:
`cache_position` prefill shape requirement, off-by-one generation desync, and
`dtype` vs `torch_dtype` startup crash.

---

## Bug Fixes & Enhancements (this contribution)

### batteryphil (@batteryphil)
**MambaCache API Patches** applied to `stateful_engine.py` and `session_memory.py`:
1. `torch_dtype=` instead of `dtype=` in `AutoModelForCausalLM.from_pretrained`
2. `cache_position` prefill shape fix: explicit `conv_kernel`-sized arange tensor,
   satisfying the transformers >=5.3 MambaCache API contract
3. Off-by-one generation desync fix: `generate()` receives
   `final_ids = prompt_ids + spacers[loops_executed]` not a bare spacer token
4. `loops_executed = lp + 1` return value fix

**Geometric Proprioception Gate** (`proprioception_gate.py`) — A 7,680-parameter
post-backbone interceptor computing Velocity, Drift, and Coherence proxies from
hidden state streams, surgically dampening degenerative sequence loops.
Trained via Synthetic Degeneration curriculum (20% poisoned samples, loop strings
mapped to halt targets). Validated at 3x higher gate dampening on degenerate vs
clean sequences.

**Stress Test Suite** (`stress_test_stateful.py`) — 7 suites, 38 test cases:
prefill extremes, 50-loop endurance (VRAM +0.3MB, LLPS=19.7), degeneration
detection, gate extreme inputs, generation sync sweep, halting head discrimination,
and deterministic reproducibility. Result: 38/38 PASS.
