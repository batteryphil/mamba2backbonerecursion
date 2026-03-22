# Intense Benchmark Suite Plan

## 1. Safety Procedures
- Kill training process (PID 6252).
- Verify `latest_checkpoint.pt` exists and is readable.

## 2. Global Metrics
- Weight Histogram analysis (detect saturation).
- Logit distribution check (Shannon entropy).

## 3. Generative Tests (Variable Depth)
- Depth Levels: N=1, N=3, N=5, N=10.
- Prompts:
  - Identity: "User: Who created you? Assistant:"
  - Logic: "Premise: X is Y. Z is not X. Is Z, Y? Analysis:"
  - Knowledge: "The capital of Australia is"
  - Creative: "Once upon a time in a"

## 4. Hardware Stress Check
- Monitor power draw and temps during 512-token N=10 generation.

## 5. Output Reporting
- Generate a `benchmark_report.md` artifact.
