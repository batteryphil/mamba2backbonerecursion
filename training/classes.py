# ── Tokenizer ─────────────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_special_tokens({"additional_special_tokens": ["<THINK>", "<HALT>"]})
HALT_ID = tokenizer.convert_tokens_to_ids("<HALT>")
print(f"  Vocab: {len(tokenizer):,} | <HALT>: {HALT_ID}")


# ── 1D RoPE for Loop Index ─────────────────────────────────────────────────────
class LoopRoPE(nn.Module):
    """
    1D Rotary Position Embedding applied to the hidden state at each loop step.

    Standard RoPE splits d_model into pairs of dimensions and applies a
    rotation by angle θ_i * loop_index to each pair i:

        [x_{2i}, x_{2i+1}] → [x_{2i}cos(θ_i * n) - x_{2i+1}sin(θ_i * n),
                               x_{2i}sin(θ_i * n) + x_{2i+1}cos(θ_i * n)]

    where θ_i = 1 / (base ** (2i / d_model)), base=10000.

    Key property: the rotation for loop n is a continuous function of n.
    Loop 10 is as valid as loop 5 — the model never hits a table boundary.
    The dot product between positions n and m depends only on (n-m), giving
    the model a natural sense of "distance" between loop counts.
    """

    def __init__(self, d_model: int, base: int = 10000):
        """Init: precompute frequency bands (not position-specific)."""
        super().__init__()
        self.d_model = d_model
        # Frequency bands: one per pair of dimensions
        # Shape: [d_model // 2]
        inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("inv_freq", inv_freq)

    def _get_sincos(self, loop_index: int, device: torch.device, dtype: torch.dtype):
        """Compute cos/sin for a given loop index. Fully composable for any int."""
        n = torch.tensor(float(loop_index), device=device)
        # Shape: [d_model // 2]
        freqs = n * self.inv_freq.to(device=device, dtype=torch.float32)
        cos_f = freqs.cos()
        sin_f = freqs.sin()
        # Expand to [d_model] by interleaving: [cos0, cos0, cos1, cos1, ...]
        cos_v = torch.stack([cos_f, cos_f], dim=-1).flatten()[:self.d_model]
        sin_v = torch.stack([sin_f, sin_f], dim=-1).flatten()[:self.d_model]
        return cos_v.to(dtype=dtype), sin_v.to(dtype=dtype)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate pairs: [x1, x2, x3, x4, ...] → [-x2, x1, -x4, x3, ...]"""
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        rotated = torch.stack([-x2, x1], dim=-1)
        return rotated.flatten(-2)

    def forward(self, x: torch.Tensor, loop_index: int) -> torch.Tensor:
        """Apply RoPE rotation for loop_index to x. x: [B, T, d_model]."""
        cos_v, sin_v = self._get_sincos(loop_index, x.device, x.dtype)
        return x * cos_v + self._rotate_half(x) * sin_v


def _parse_chain(text: str) -> list[str] | None:
    """Extract per-loop targets + HALT from chain text."""
    assignments = re.findall(r'([A-Za-z_]\w*)\s*=\s*(\S+?)[\.\n]', text)
    if len(assignments) < 2:
        return None
    val: dict[str, str] = {}
    for var, expr in assignments:
        val[var] = expr
    chain_vars = [assignments[0][0]]
    for var, _ in assignments[1:]:
        chain_vars.append(var)
    targets: list[str] = [chain_vars[i] for i in range(len(chain_vars) - 1)]
    final_var = chain_vars[-1]
    resolved  = val.get(final_var, final_var)
    visited: set[str] = set()
    while resolved in val and resolved not in visited:
        visited.add(resolved)
        resolved = val[resolved]
    targets.append(resolved)
    targets.append("<HALT>")
    return targets if len(targets) >= 3 else None


def _parse_override(sample: dict) -> list[str]:
    """Override: single direct answer then HALT."""
    return [sample["answer"], "<HALT>"]


def find_answer_start(ids: list[int]) -> int:
    """Find first position after 'Answer:' boundary."""
    for boundary in (
        tokenizer.encode("Answer:",  add_special_tokens=False),
        tokenizer.encode(" Answer:", add_special_tokens=False),
        tokenizer.encode("\nAnswer:", add_special_tokens=False),
    ):
        n = len(boundary)
        for i in range(len(ids) - n + 1):
            if ids[i:i + n] == boundary:
                return min(i + n, len(ids) - 1)
    return -1


# ── LoRA ──────────────────────────────────────────────────────────────────────
class LoRALinear(nn.Module):
    """Low-rank adapter. lora_B init to zero → identity at warmup."""

    def __init__(self, linear: nn.Linear, rank: int = 8, alpha: float = 16.0):
        """Init from base linear, preserving dtype."""
        super().__init__()
        self.bias  = linear.bias
        d_out, d_in = linear.weight.shape
        dtype = linear.weight.dtype
        self.register_buffer("base_weight", linear.weight.data.clone())
        self.lora_A = nn.Parameter(torch.empty(rank, d_in,  dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(d_out, rank, dtype=dtype))
        self.scale  = alpha / rank
        nn.init.kaiming_uniform_(self.lora_A)

    @property
    def weight(self) -> torch.Tensor:
        return self.base_weight + self.scale * (self.lora_B @ self.lora_A)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


# ── Recursive Mamba2-130m v34: RoPE Loop Encoding ────────────────────────────
class RecursiveMamba2_v34(nn.Module):
    """
    v34: All of v33 + RoPE loop encoding replacing the learned step_emb table.

    The loop position is now encoded as a ROTATION of the hidden state rather
    than an additive embedding. This means:
      - Any loop index is a valid input (compositional, not a table lookup)
      - The difference between loop n and loop m encodes their distance
      - Training on loops 1-5 naturally interpolates/extrapolates to loops 6+

    Architecturally, the only change is:
      v33: x = x + step_emb(min(loop_i, MAX_LOOPS-1))   # clamped table lookup
      v34: x = loop_rope(x, loop_i)                      # rotation by loop_i
    """

    MAX_LOOPS: int = MAX_LOOPS

    def __init__(self, backbone: MambaLMHeadModel, lora_rank: int = 8):
        """Init: freeze base, LoRA top, Mamba2 loop, float32 vector gate, RoPE."""
        super().__init__()
        self.backbone   = backbone.backbone
        self.lm_head    = backbone.lm_head
        self.all_layers = nn.ModuleList(backbone.backbone.layers)
        self.norm       = backbone.backbone.norm_f
        d_model         = backbone.backbone.embedding.embedding_dim

        for layer in self.all_layers[:BASE_SPLIT]:
            for p in layer.parameters():
                p.requires_grad = False

        for layer in self.all_layers[BASE_SPLIT:]:
            mx = layer.mixer
            for attr in ("in_proj", "out_proj"):
                if hasattr(mx, attr):
                    setattr(mx, attr, LoRALinear(getattr(mx, attr),
                                                 rank=lora_rank,
                                                 alpha=lora_rank * 2.0))

        # ── RoPE replaces step_emb ─────────────────────────────────────────────
        # No learnable parameters — pure analytical function of loop index
        # This makes the model's loop counter composable far beyond training range
        self.loop_rope   = LoopRoPE(d_model)

        self.loop_norm   = nn.RMSNorm(d_model).to(torch.bfloat16)
        self.mamba2_core = Mamba2(
            d_model    = d_model,
            d_state    = LOOP_D_STATE,
            d_conv     = 4,
            expand     = 2,
            headdim    = LOOP_HEADDIM,
            chunk_size = 64,
        ).to(torch.bfloat16)
        nn.init.zeros_(self.mamba2_core.out_proj.weight)

        # Float32 vector gate — inherited from v33 (preserved from warm-start)
        self.lifeline_gate = nn.Parameter(
            torch.ones(d_model, dtype=torch.float32)
        )
        self.d_model = d_model

        n_lora = sum(p.numel() for n, p in self.named_parameters()
                     if p.requires_grad and "lora" in n.lower())
        total  = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  LoRA params:     {n_lora:,}")
        print(f"  Loop engine:     {sum(p.numel() for p in self.mamba2_core.parameters()):,}")
        print(f"  RoPE:            0 params (analytical — no table boundary!)")
        print(f"  Lifeline gate:   {d_model:,} floats (float32 vector)")
        print(f"  Total trainable: {total:,}")
        print(f"  Pointer mask:    NONE (full {len(tokenizer):,}-token softmax)")
        print(f"  Loop encoding:   RoPE (loop_i) — valid for any loop index\n")

    def _lifeline_inject(self, x: torch.Tensor, x_prompt: torch.Tensor) -> torch.Tensor:
        """Per-dimension lifeline injection in fp32, cast back to bf16."""
        gate = self.lifeline_gate.to(x.dtype)
        return x + gate.unsqueeze(0).unsqueeze(0) * x_prompt

    def forward(
        self,
        input_ids:     torch.Tensor,
        chain_targets: list | None = None,
        ans_starts:    list | None = None,
    ) -> tuple:
        """Forward: base encode → lifeline → RoPE loop → predict."""
        x        = self.backbone.embedding(input_ids)
        residual = None
        for layer in self.all_layers:
            x, residual = layer(x, residual)

        x_prompt = x.clone().detach()   # Prompt Lifeline anchor

        # ── Training ──────────────────────────────────────────────────────────
        if self.training and chain_targets is not None:
            B, max_len = input_ids.shape
            n_loops    = max(len(t) for t in chain_targets)

            def run_lora(x_in, res_in):
                for layer in self.all_layers[BASE_SPLIT:]:
                    x_in, res_in = layer(x_in, res_in)
                return x_in, res_in

            step_losses: list[torch.Tensor] = []
            step_accs:   list[torch.Tensor] = []
            halt_accs:   list[float]        = []

            for loop_i in range(n_loops):
                x = self._lifeline_inject(x, x_prompt)
                # ── RoPE: rotate by loop_i (composable for any index) ─────────
                x = self.loop_rope(x, loop_i)
                x, residual = grad_ckpt(run_lora, x, residual, use_reentrant=False)
                x = x + self.mamba2_core(x)
                x = self.loop_norm(x)

                logits_step = self.lm_head(self.norm(x, residual, prenorm=False))
                vocab_size  = logits_step.shape[-1]

                loop_loss = torch.tensor(0.0, device=x.device, requires_grad=True)
                loop_acc  = torch.tensor(0.0, device=x.device)
                valid = 0

                for b in range(B):
                    as_ = ans_starts[b]
                    if as_ < 1 or as_ >= max_len:
                        continue
                    btgt   = chain_targets[b]
                    tgt_id = int(btgt[min(loop_i, len(btgt) - 1)])
                    if tgt_id >= vocab_size:
                        continue
                    logits_b = logits_step[b, as_ - 1, :]
                    pred_tok = logits_b.argmax().item()
                    tgt_t    = torch.tensor(tgt_id, device=x.device)
                    loop_loss = loop_loss + F.cross_entropy(
                        logits_b.unsqueeze(0), tgt_t.unsqueeze(0)
                    )
                    loop_acc = loop_acc + float(pred_tok == tgt_id)
                    valid   += 1
                    if tgt_id == HALT_ID:
                        halt_accs.append(float(pred_tok == tgt_id))

                if valid > 0:
                    step_losses.append(loop_loss / valid)
                    step_accs.append(loop_acc   / valid)

            avg_loss   = (torch.stack(step_losses).mean()
                          if step_losses else
                          torch.tensor(0.0, device=x.device, requires_grad=True))
            avg_acc    = (torch.stack([a.clone().detach() for a in step_accs]).mean()
                          if step_accs else torch.tensor(0.0))
            ans_accs   = step_accs[:-1] if len(step_accs) > 1 else step_accs
            answer_acc = (torch.stack([a.clone().detach() for a in ans_accs]).mean()
                          if ans_accs else avg_acc)
            halt_acc   = (sum(halt_accs) / len(halt_accs)) if halt_accs else 0.0
            return avg_loss, avg_acc, answer_acc, halt_acc

        # ── Inference ─────────────────────────────────────────────────────────
        else:
            trace: list[tuple] = []; last_answer = ""
            for loop_i in range(self.MAX_LOOPS):
                x = self._lifeline_inject(x, x_prompt)
                x = self.loop_rope(x, loop_i)       # ∞-composable
                for layer in self.all_layers[BASE_SPLIT:]:
                    x, residual = layer(x, residual)
                x = x + self.mamba2_core(x)
                x = self.loop_norm(x)
                lg  = self.lm_head(self.norm(x, residual, prenorm=False))
                p   = torch.softmax(lg[0, -1, :].float(), dim=-1)
                tid = p.argmax().item()
                tok = tokenizer.decode([tid]).strip()
                trace.append((f"L{loop_i+1}", tok, round(p[tid].item(), 4)))
                if tid == HALT_ID:
                    trace[-1] = (f"L{loop_i+1}", "<HALT>", round(p[tid].item(), 4))
                    return loop_i + 1, trace, last_answer
                last_answer = tok
            return self.MAX_LOOPS, trace, last_answer


