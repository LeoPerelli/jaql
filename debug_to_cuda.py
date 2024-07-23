# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import quantise_model,get_model_memory_size, compute_quantisation_mse, Perplexity
from datasets import load_dataset
import torch

torch.cuda.memory._record_memory_history()

tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")


model = model.transformer.wte
inputs = torch.ones((1,10000), dtype=torch.long, device='cuda')
model._tied_weights_keys = []
model, parameter_mapping = quantise_model(model, chunk_size=1024)
model = model.to('cuda')

with torch.no_grad():
    out = model(inputs)
    out = 2*out
torch.cuda.memory._dump_snapshot("int8_to_cuda_emb.pickle")