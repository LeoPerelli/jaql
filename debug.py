# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import quantise_model,get_model_memory_size, compute_quantisation_mse, Perplexity
from datasets import load_dataset
import torch

torch.cuda.memory._record_memory_history()



tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")

# model = model.half()

text = [t for t in load_dataset("wikitext", "wikitext-2-raw-v1", split="test")['text'] if len(t) > 30]
get_model_memory_size(model)
model, parameter_mapping = quantise_model(model, chunk_size=1024)
get_model_memory_size(model, parameter_mapping)

text = text[0:2]
perplexity = Perplexity()
p = perplexity._compute(text, model, tokenizer)
print(p['mean_perplexity'])


torch.cuda.memory._dump_snapshot("int8_1024_del_state.pickle")