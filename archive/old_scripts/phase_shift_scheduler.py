import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

class PhaseShiftLRScheduler:
    def __init__(self, optimizer, start_lr=4e-5, min_lr=1e-6, decay_steps=1000, burn_in_steps=150):
        """
        A custom scheduler that waits for N=3 to trigger, allows a burn-in 
        period for the loss spike to settle, and then initiates a cosine decay.
        Supports SGDR-style Warm Restarts: pass warm_restart() to reset the cycle.
        """
        self.optimizer = optimizer
        self.start_lr = start_lr
        self.min_lr = min_lr
        self.decay_steps = decay_steps
        self.burn_in_steps = burn_in_steps
        
        # We initialize the PyTorch scheduler, but we won't step it until triggered
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.decay_steps, eta_min=self.min_lr)
        
        self.n3_trigger_step = None
        self.is_cooling_down = False

    def warm_restart(self, current_step: int):
        """Perform an SGDR warm restart: reset LR to start_lr and re-arm cosine decay countdown."""
        print(f"\n🔥 [LR SCHEDULER] SGDR Warm Restart at Step {current_step}! LR reset to {self.start_lr:.2e}")
        for pg in self.optimizer.param_groups:
            pg['lr'] = self.start_lr
        # Re-create the cosine annealing schedule from scratch
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.decay_steps, eta_min=self.min_lr)
        # Re-arm the trigger so it starts cosine decay immediately (no burn-in on restart)
        self.n3_trigger_step = current_step - self.burn_in_steps
        self.is_cooling_down = False

    def step(self, current_step, current_n_depth):
        """Advance the scheduler by one step."""
        # 1. Detect the exact moment N=3 is triggered
        if current_n_depth == 3 and self.n3_trigger_step is None:
            self.n3_trigger_step = current_step
            print(f"\n⚙️ [LR SCHEDULER] N=3 Detected at Step {current_step}. Starting {self.burn_in_steps}-step burn-in...")

        # 2. Wait for the burn-in period to finish so the gradient shockwave settles
        if self.n3_trigger_step is not None:
            steps_since_n3 = current_step - self.n3_trigger_step
            
            if steps_since_n3 >= self.burn_in_steps:
                if not self.is_cooling_down:
                    print(f"\n❄️ [LR SCHEDULER] Burn-in complete. Initiating Cosine Annealing Cooldown.")
                    self.is_cooling_down = True
                
                # 3. Step the cosine scheduler
                self.scheduler.step()
                
        # Return the current learning rate for your telemetry logs
        return self.optimizer.param_groups[0]['lr']
