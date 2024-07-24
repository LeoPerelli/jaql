# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import quantise_model,get_model_memory_size, compute_quantisation_mse, Perplexity
from datasets import load_dataset
import torch
from tqdm import tqdm


def compute_perplexity(chunk_size = None):

    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")

    # model = model.half()

    text = [t for t in load_dataset("wikitext", "wikitext-2-raw-v1", split="test")['text'] if len(t) > 30]

    if chunk_size:
        get_model_memory_size(model)
        model, parameter_mapping = quantise_model(model, chunk_size=chunk_size)
        get_model_memory_size(model, parameter_mapping)


    perplexity = Perplexity()
    p = perplexity._compute(text, model, tokenizer, batch_size=16)

    del model
    torch.cuda.empty_cache()
    return p['mean_perplexity']

# torch.cuda.memory._record_memory_history()
# torch.cuda.memory._dump_snapshot("int8_1024_del_state_real.pickle")

d = []
for chunk in [32, None]: # 64, 128, 256, 1024]:

    d.append(compute_perplexity(chunk_size=chunk))

print(d)


