# Project Phoenix: V3.2 Post-Mortem & V4 Architectural Laws

## Executive Summary
DiM-LLM v3.2 demonstrated exceptional engine stability and throughput (~10k TPS) thanks to the Mamba-Diffusion backbone. However, the model reached a "Semantic Ceiling" caused by representational entanglement and sampling timidity. V4 adopts the **Clean Slate Protocol** to resolve these fundamental flaws.

---

## Law I: The Engine Bedrock
Stability is the prerequisite for intelligence. The following configurations are non-negotiable for high-dimensional convergence:
- **Compute:** Linux TF32/OMP tuning is mandatory for numerical precision.
- **Initialization:** Zero-Init AdaLN (Adaptive Layer Normalization) ensures the model starts as an identity map, preventing early-layer gradient explosion.
- **Optimization:** AdaGC (Adaptive Gradient Clipping) with `lambda_factor=0.02` is the primary defense against NaN divergence during high-learning-rate phasic transitions.

## Law II: The Data Trap (Representational Entanglement)
*Problem:* Concentrating Chain-of-Thought (CoT) logic exclusively within JSON structures led to **catastrophic mode collapse**. The model identified "Reasoning" as a subset of "Syntax," losing the ability to generalize logic outside of bracketed delimiters.
*Solution:* **Multi-Surface Logic Grounding.** Logic problems must be presented in diverse formats (JSON, Python, Lists, Plain Text) to force the latent space to decouple semantic reasoning from syntactic constraints.

## Law III: The Sampler Trap (The Mask Trap)
*Problem:* Standard cosine/linear mask schedules allow the model to remain in a state of "probabilistic indecision" until the final steps. At low temperatures, the model refuses to unmask difficult tokens, essentially [MASK]-hiding.
*Solution:* **The Commitment Cliff.** The mask ratio must hit $0.0$ at the 85% mark of the inference steps ($t \leq 0.15$). This forces the model to spend the final 15% of compute power polishing and denoising committed tokens, ensuring structural integrity.

---

**Protocol Status:** V4 Initialization Authorized.
**Sign-off:** Lead Systems Architect & Forensic ML Engineer
