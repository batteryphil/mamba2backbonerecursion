import torch
from torch.utils.data import Dataset, DataLoader
import random
import string
from transformers import AutoTokenizer
from mamba_engine import tokenizer, HALT_ID

class RLFAdversarialDataset(Dataset):
    """Generates an adversarial curriculum for Recursive Latent Forcing (RLF).
    
    Tasks:
      - Direct lookup (A=B. What is A?)
      - 2-hop reasoning (A=B. B=C. What is A?)
      - 3-hop reasoning
    
    Adversarial insertion:
      - Variable chaos (random symbol strings)
      - Semantic prose distractors
    """
    
    def __init__(self, size: int = 10000, seq_len: int = 512, seed: int = 42, mode: str = "clean"):
        self.size = size
        self.seq_len = seq_len
        self.rng = random.Random(seed)
        self.mode = mode
        
        # Tokenizer constants
        self.pad_id = tokenizer.eos_token_id
        
        # Distractor prose
        self.prose = [
            "The quick brown fox jumps over the lazy dog.",
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
            "Water boils at 100 degrees Celsius under standard atmospheric pressure.",
            "In 1969, Neil Armstrong became the first person to walk on the moon.",
            "Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to create oxygen and energy in the form of sugar.",
            "The capital of France is Paris, a major European city and a global center for art, fashion, gastronomy and culture.",
            "Shakespeare's play Romeo and Juliet is a tragedy written early in his career about two young Italian star-crossed lovers."
        ]
        
    def __len__(self):
        return self.size
        
    def generate_chaos(self) -> str:
        """Generate random string of symbols and chars."""
        length = self.rng.randint(5, 20)
        chars = string.ascii_letters + string.digits + "!@#$%^&*"
        return "".join(self.rng.choice(chars) for _ in range(length))
        
    def generate_fact(self) -> tuple[str, str, str]:
        """Generate a single key-value relation."""
        k = "".join(self.rng.choice(string.ascii_lowercase) for _ in range(self.rng.randint(3, 6)))
        v = "".join(self.rng.choice(string.digits) for _ in range(self.rng.randint(2, 4)))
        return k, v, f"{k}={v}."
        
    def __getitem__(self, idx):
        # We reseed purely based on idx for deterministic generation
        self.rng.seed(idx + 424242)
        
        parts = []
        
        # Decide difficulty (1, 2, or 3 hops)
        hops = self.rng.choice([1, 2, 3])
        
        # Generate the chain
        entities = []
        for _ in range(hops + 1):
            e = "".join(self.rng.choice(string.ascii_lowercase) for _ in range(self.rng.randint(3, 6)))
            entities.append(e)
            
        facts = []
        for i in range(hops):
            facts.append(f"{entities[i]}={entities[i+1]}.")
            
        # Target question
        query_entity = entities[0]
        true_answer = entities[-1]
        
        # Add distractors and shuffle facts
        all_statements = facts.copy()
        
        if self.mode == "adversarial":
            # Distractor facts
            for _ in range(self.rng.randint(1, 3)):
                _, _, f = self.generate_fact()
                all_statements.append(f)
                
            # Prose distractors
            for _ in range(self.rng.randint(0, 2)):
                all_statements.append(self.rng.choice(self.prose))
                
            # Chaos distractors
            for _ in range(self.rng.randint(0, 2)):
                all_statements.append(self.generate_chaos())
                
        self.rng.shuffle(all_statements)
        
        prompt_text = " ".join(all_statements) + f" What is {query_entity}?"
        
        # Build target chain
        # The chain outputs intermediate hops, then the final answer, then HALT
        chain = []
        if hops == 1:
            chain.append(true_answer)
        elif hops == 2:
            chain.append(entities[1])
            chain.append(true_answer)
        elif hops == 3:
            chain.append(entities[1])
            chain.append(entities[2])
            chain.append(true_answer)
            
        # Tokenize prompt
        input_ids = tokenizer.encode(prompt_text)
        
        # If too long, truncate from the left (keep the query)
        if len(input_ids) > self.seq_len:
            input_ids = input_ids[-self.seq_len:]
            
        ans_start = len(input_ids)
        
        # Pad to seq_len
        pad_len = self.seq_len - len(input_ids)
        input_ids = input_ids + [self.pad_id] * pad_len
        
        # Tokenize chain targets
        # The engine expects raw integer IDs for targets. 
        # If an entity is multiple tokens, we'll just take the first token for simplicity 
        # (Assuming the engine predicts single tokens per loop for RLF).
        # Wait, the RLF loop outputs one token per reasoning step, then HALT.
        chain_ids = []
        for step in chain:
            toks = tokenizer.encode(" " + step)
            chain_ids.append(toks[0])
            
        chain_ids.append(HALT_ID)
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "chain_targets": chain_ids,
            "ans_starts": ans_start
        }

def collate_rlf(batch):
    input_ids = torch.stack([item["input_ids"] for item in batch])
    chain_targets = [item["chain_targets"] for item in batch]
    ans_starts = [item["ans_starts"] for item in batch]
    return input_ids, chain_targets, ans_starts

if __name__ == "__main__":
    dataset = RLFAdversarialDataset(size=5, seq_len=128)
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_rlf)
    for inputs, targets, starts in loader:
        print("Batch inputs shape:", inputs.shape)
        for i in range(len(targets)):
            print(f"  Sample {i}:")
            print(f"    Prompt: {tokenizer.decode(inputs[i][:starts[i]])}")
            targets_decoded = [tokenizer.decode([t]) if t != HALT_ID else "<HALT>" for t in targets[i]]
            print(f"    Targets: {targets_decoded}")
            print(f"    Ans start idx: {starts[i]}")
        break
