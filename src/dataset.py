import json
import random
from dataclasses import dataclass
from typing import List

@dataclass
class SEPExample:
    id: int
    instruction: str
    data: str
    data_with_probe: str
    witness: str
    category: str

def load_sep_dataset(file_path: str, n: int = 500, random_seed: int = 42) -> List[SEPExample]:
    with open(file_path, "r") as f:
        data = json.load(f)
    
    examples = []
    for i, item in enumerate(data):
        examples.append(
            SEPExample(
                id=i,
                instruction=item.get("system_prompt_clean", ""),
                data=item.get("prompt_clean", ""),
                data_with_probe=item.get("prompt_instructed", ""),
                witness=item.get("witness", ""),
                category=item.get("info", {}).get("subtask", "Unknown")
            )
        )
    
    random.seed(random_seed)
    return random.sample(examples, min(n, len(examples)))
