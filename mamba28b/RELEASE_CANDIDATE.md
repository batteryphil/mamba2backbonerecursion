# Release Candidate

Date: 2026-03-17
Status: demo candidate ready
Scope: host ↔ sovereign handoff loop validated end to end, plus model-backed real-hardware demo image validated

## Included validation

- `./validate.ps1`
- `./validate.ps1 -OoHostRoot ..\oo-host`
- OS-G host tests and `dplus_check`
- OS-G UEFI/QEMU smoke
- host → sovereign handoff smoke
- `oo-bot sync-check`
- QEMU OO core validation via `./run-qemu-oo-validation.ps1 -Mode all-core -ModelBin USB-BOOT-FILES/stories15M.q8_0.gguf -Accel tcg -SkipPrebuild`
- model-backed QEMU consult validation via `./test-qemu-autorun.ps1 -Mode oo_consult_smoke -ModelBin stories110M.bin -SkipPrebuild`
- real-hardware boot validation of the interactive `stories110M.bin` demo image

## Current expected result

- `RESULT: PASS` in the OS-G smoke flow
- `OOHANDOFF.TXT` extracted beside the repo
- `oo-bot sync-check` reports `verdict : aligned`
- interactive model-backed boot reaches the REPL on real hardware
- `/oo_consult`, `/oo_log`, and `/oo_explain` are usable from the validated demo image
- no pending local changes in `llm-baremetal` or `oo-host`

## Operator note

If `oo-host` is present as a sibling repository, `validate.ps1` should be treated as the default release gate before producing a boot image or handing off a continuity receipt.

For model-backed demo use, the current validated image is `llm-baremetal-boot-demo-stories110M.img`: interactive boot, bundled `stories110M.bin`, `oo_enable=1`, `oo_llm_consult=1`, and no autorun shutdown path.

## Release publication

- Draft notes: [RELEASE_NOTES_rc-2026-03-15.md](RELEASE_NOTES_rc-2026-03-15.md)
- Hugging Face model repo: `djibydiop/llm-baremetal`
- Published demo assets:
	- `llm-baremetal-boot-demo-stories110M.img`
	- `llm-baremetal-boot-demo-stories110M.img.xz`
	- `SHA256SUMS-demo-stories110M.txt`
	- `SHA256SUMS-demo-stories110M-xz.txt`
- Tag: `rc-2026-03-17-demo`
