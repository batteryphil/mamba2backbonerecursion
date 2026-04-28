# Release Notes — rc-2026-03-17-demo

Date: 2026-03-17
Tag: `rc-2026-03-17-demo`
Status: draft for GitHub Release publication

## Highlights

- End-to-end host ↔ sovereign handoff flow validated.
- `oo-host` operator tooling hardened with explicit tests, CI checks, and artifact rendering.
- `llm-baremetal` validation now treats handoff + `sync-check` as a stable release gate when a sibling `oo-host` workspace is present.
- Relative `-OoHostRoot` overrides are now resolved robustly for sibling-clone workflows.
- A model-backed interactive demo image with bundled `stories110M.bin` has been validated on real hardware.
- The interactive OO consult path (`oo_enable=1`, `oo_llm_consult=1`) has been published as a ready-to-flash demo artifact.

## What is included in this candidate

- x86_64 UEFI no-model boot image flow
- OS-G host tests and policy verification
- OS-G UEFI/QEMU smoke validation
- host → sovereign receipt extraction via `OOHANDOFF.TXT`
- aligned host/export/receipt verification via `oo-bot sync-check`
- QEMU OO outcome / reboot / handoff / consult core matrix
- interactive real-hardware `stories110M.bin` demo image with OO consult enabled

## Validation summary

- `./validate.ps1` passes
- `./validate.ps1 -OoHostRoot ..\oo-host` passes
- OS-G smoke reports `RESULT: PASS`
- handoff smoke reports `PASS`
- `oo-bot sync-check` reports `verdict               : aligned`
- `./run-qemu-oo-validation.ps1 -Mode all-core -ModelBin USB-BOOT-FILES/stories15M.q8_0.gguf -Accel tcg -SkipPrebuild` passes
- `./test-qemu-autorun.ps1 -Mode oo_consult_smoke -ModelBin stories110M.bin -SkipPrebuild` passes
- the validated demo image boots to the interactive REPL on real hardware

## Suggested release assets

- `llm-baremetal-boot-nomodel-x86_64.img.xz`
- `SHA256SUMS.txt`
- `llm-baremetal-boot-demo-stories110M.img`
- `llm-baremetal-boot-demo-stories110M.img.xz`
- `SHA256SUMS-demo-stories110M.txt`
- `SHA256SUMS-demo-stories110M-xz.txt`
- optional operator artifact bundle generated from `oo-host handoff-pack`

## Operator guidance

- This candidate does not bundle model weights.
- Keep `oo-host` adjacent to `llm-baremetal` when running the full release gate locally.
- Treat [RELEASE_CANDIDATE.md](RELEASE_CANDIDATE.md) as the short status page and this document as the publishable release draft.
- For the validated demo flow, flash `llm-baremetal-boot-demo-stories110M.img` and use `/cfg`, `/diag`, `hi`, `/oo_status`, and `/oo_consult` as the short live sequence.

## Known constraints

- Hardware virtualization is still reported disabled on the current validation host, so QEMU validation is running in a compatible non-accelerated path.
- Both raw `.img` and compressed `.img.xz` demo artifacts are now published; prefer the compressed artifact for distribution and keep the raw image for direct local flashing workflows.