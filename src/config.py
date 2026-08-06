import os
from dotenv import load_dotenv

load_dotenv()

MODELS = {
    "qwen14b": "qwen2.5:14b",
    "local_llama": "llama3.1:latest",
    "hermes-llama": "NousResearch/Hermes-2-Pro-Llama-3-8B",
    "llama-3.1": "llama3.1",
    "mistral-nemo": "mistral-nemo:latest",
    "qwen-3b": "qwen2.5:3b"
}
