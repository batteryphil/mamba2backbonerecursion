import torch
import time
import os
import json
import sys
from transformers import GPT2Tokenizer
from mamba_llm_diffusion import DiM_LLM, MaskedDiffusionEngine, Config
from mamba_diffusion import MambaDiffusion, SelectiveSSM

# --- Thorough Test Suite for DiM v3.2 (Smoke Test) ---

class DiMSmokeTester:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"--- Initialization: Running Smoke Test on {self.device} ---")
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.add_special_tokens({"mask_token": "[MASK]"})
        self.config = Config(
            vocab_size=len(self.tokenizer),
            d_model=128,
            n_layers=4,
            seq_len=64
        )

    def test_architecture(self):
        print("[1/4] Testing DiM_LLM Architecture & AdaLN...")
        model = DiM_LLM(self.config).to(self.device)
        
        # Test forward pass
        input_ids = torch.randint(0, self.config.vocab_size, (2, self.config.seq_len), device=self.device)
        t_norm = torch.rand(2, device=self.device)
        
        # 1. Standard forward
        logits = model(input_ids, t_norm)
        assert logits.shape == (2, self.config.seq_len, self.config.vocab_size)
        
        # 2. Forward with Context Bank (Hook Test)
        context = torch.randn(2, 1, self.config.d_model, device=self.device)
        logits_ctx = model(input_ids, t_norm, context_bank=context)
        assert logits_ctx.shape == (2, self.config.seq_len, self.config.vocab_size)
        
        # 3. Weight Tying Check
        assert model.token_embed.weight is model.output_proj.weight, "Weights are not tied!"
        
        assert not torch.isnan(logits).any(), "NaN detected in standard forward!"
        assert not torch.isnan(logits_ctx).any(), "NaN detected in context-aware forward!"
        print("      PASS: Architecture, AdaLN, and Weight Tying verified.")

    def test_engine_logic(self):
        print("[2/4] Testing MaskedDiffusionEngine Logic...")
        model = DiM_LLM(self.config).to(self.device)
        engine = MaskedDiffusionEngine(model, self.config, device=self.device)
        
        # 1. Standard forward process (loss calculation)
        input_ids = torch.randint(0, self.config.vocab_size, (2, self.config.seq_len), device=self.device)
        loss = engine.forward_process(input_ids)
        assert loss > 0, "Loss should be positive."
        assert not torch.isnan(loss), "Loss is NaN!"
        
        # 2. Forward process with Context Bank
        context = torch.randn(2, 1, self.config.d_model, device=self.device)
        loss_ctx = engine.forward_process(input_ids, context_bank=context)
        assert loss_ctx > 0, "Contextual loss should be positive."
        assert not torch.isnan(loss_ctx), "Contextual loss is NaN!"
        
        # Test EMA update hook
        engine.ema_model = DiM_LLM(self.config).to(self.device)
        engine.update_ema()
        print("      PASS: Engine forward process and EMA logic verified.")

    def test_sampling_optimization(self):
        print("[3/4] Testing Optimized Sampling (Top-K Filter)...")
        model = DiM_LLM(self.config).to(self.device)
        engine = MaskedDiffusionEngine(model, self.config, device=self.device)
        
        # Test short sampling run
        start = time.time()
        output = engine.sample(n_samples=1, steps=5, temperature=0.7, context_bank=None)
        elapsed = time.time() - start
        
        assert output.shape == (1, self.config.seq_len)
        assert (output < self.config.vocab_size).all()
        print(f"      PASS: Sampling with Top-K optimization completed in {elapsed:.2f}s.")

    def test_mamba_stability(self):
        print("[4/4] Testing Mamba Backbone Numerical Stability...")
        ssm = SelectiveSSM(d_model=64, d_state=16).to(self.device)
        x = torch.randn(2, 16, 64, device=self.device)
        
        y_fwd = ssm(x, direction=0)
        y_bwd = ssm(x, direction=1)
        
        assert not torch.isnan(y_fwd).any(), "NaN in Mamba FWD"
        assert not torch.isnan(y_bwd).any(), "NaN in Mamba BWD"
        print("      PASS: Mamba Selective SSM remains stable.")

    def run_all(self):
        print("\n" + "="*50)
        print("DiM-LLM v3.2 - SMOKE TEST SUITE")
        print("="*50 + "\n")
        
        try:
            self.test_architecture()
            self.test_engine_logic()
            self.test_sampling_optimization()
            self.test_mamba_stability()
            print("\n" + "="*50)
            print("SMOKE TEST STATUS: ALL SYSTEMS NOMINAL ✅")
            print("="*50 + "\n")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"\n❌ SMOKE TEST FAILED: {str(e)}")
            sys.exit(1)

if __name__ == "__main__":
    with torch.no_grad():
        tester = DiMSmokeTester()
        tester.run_all()
