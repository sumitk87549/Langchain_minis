from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load the tokenizer and model from Hugging Face
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct", cache_dir="../models/llama3.1")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct", torch_dtype=torch.float16, device_map="auto", cache_dir="../models/llama3.1")