from dataclasses import dataclass
from evaluate import logging
from tqdm import tqdm
import evaluate
import torch
import math
import numpy as np
from torch.nn import CrossEntropyLoss

def scalar_quantisation(t, q_a=-127, q_b=127):

    assert len(t) > 1, "Bad idea to quantise a single element tensor"

    a = t.min()
    b = t.max()

    s = (q_b - q_a) / (b - a)
    q_c = (q_a + q_b) / 2
    c = (b + a) / 2
    zero_point = q_c - s * c

    x_q = (s * t + zero_point).type(torch.int8)

    return x_q, s, zero_point


def scalar_dequantisation(t_q, s, zero_point):

    assert s > 0, "Scale is not accetpable"

    t_approx = (t_q - zero_point) / s

    return t_approx


def compute_quantisation_mse(t, t_approx):

    mse = torch.sqrt(torch.square(t - t_approx).sum())
    norm = torch.sqrt(torch.square(t).sum())

    print(f"MSE: {mse}\nNorm: {norm}\nRelativeMSE: {round(float(mse/norm * 100),5)}%")

    return mse


def get_tensor_memory_size(t):

    return t.nelement() * t.element_size()

def get_model_memory_size(model, parameter_mappings=None):
    size_model = 0
    for param in model.parameters():
        size_model += get_tensor_memory_size(param)
    
    if parameter_mappings:
        for metadata_dict in parameter_mappings.values():
            size_model += get_tensor_memory_size(metadata_dict['scales'])
            size_model += get_tensor_memory_size(metadata_dict['locations'])

    print(f"model size: {size_model} / bit | {size_model / 1e6:.2f} / MB")


def quantise_tensor(t, chunk_size=512):

    shape = t.shape
    t_flat = t.flatten()
    n_chunks = math.ceil(len(t_flat) / chunk_size)
    scales = torch.zeros(n_chunks)
    locations = torch.zeros(n_chunks)

    for chunk_id in range(n_chunks):

        left = chunk_id * chunk_size
        right = min(len(t_flat), (chunk_id + 1) * chunk_size)

        t_flat[left:right], scales[chunk_id], locations[chunk_id] = scalar_quantisation(t_flat[left:right])

    t_flat = t_flat.reshape(shape)
    t_flat = t_flat.type(torch.int8)

    return t_flat, scales, locations


def dequantise_tensor(t_q, scales, locations, chunk_size):

    shape = t_q.shape
    t_q = t_q.flatten()

    n_chunks = len(scales)
    t_q = t_q.type(torch.float32)

    for chunk_id in range(n_chunks):
        left = chunk_id * chunk_size
        right = min(len(t_q), (chunk_id + 1) * chunk_size)

        t_q[left:right] = scalar_dequantisation(t_q[left:right], scales[chunk_id], locations[chunk_id])

    t_q = t_q.reshape(shape)
    return t_q

def quantise_model(model, chunk_size):

    parameter_mapping = {}    
    for parameter_name, p in tqdm(list(model.named_parameters()), desc = "Quantising model layers"):
        p_q, scales, locations = quantise_tensor(p.clone(), chunk_size)
        p.data.copy_(p_q)
        p.requires_grad = False
        p.data = p.data.to(torch.int8)
        parameter_mapping[parameter_name] = {'scales':scales, 'locations': locations, 'chunk_size': chunk_size}

    leaf_modules = get_leaf_modules(model)
    update_parameter_mappings_with_tied_parameters(model, parameter_mapping)
    apply_quantisation_hooks(model, leaf_modules, parameter_mapping)

    return model, parameter_mapping

def update_parameter_mappings_with_tied_parameters(model, parameter_mapping):

    if model._tied_weights_keys == []:
        return {}

    pointers = {}
    for module_name, module in model.named_modules():
        if module_name == '':
            continue
        for parameter_name, p in module.named_parameters():
            pointers[f'{module_name}.{parameter_name}'] = p.data_ptr()

    for parameter_name in model._tied_weights_keys:
        pointer = pointers[parameter_name]
        for p_name, p_pointer in pointers.items():
            if p_pointer == pointer:
                # copy the parameter info for the tied parameter
                parameter_mapping[parameter_name] = parameter_mapping[p_name]
                break

def get_leaf_modules(model):
    
    leaf_modules = []
    for module_name, m in model.named_modules():
        if len(list(m.named_modules())) == 1:
            leaf_modules.append(module_name)
    
    return leaf_modules

def hook_factory(leaf_module_name, parameter_mapping):

    def dequantise_hook(module, args):
        module.quantised_state = {}
        parameter_names = [p for p, _ in module.named_parameters()]
        for parameter_name in parameter_names:
            global_parameter_name = f'{leaf_module_name}.{parameter_name}'
            p = module.get_parameter(parameter_name)
            module.quantised_state[parameter_name] = p.clone()
            p_approx = dequantise_tensor(p, parameter_mapping[global_parameter_name]['scales'], parameter_mapping[global_parameter_name]['locations'],parameter_mapping[global_parameter_name]['chunk_size'])
            p.data = p.data.to(torch.float32)
            p.copy_(p_approx)

    def cleanup_hook(module, args, output):
        for parameter_name, p in module.quantised_state.items():
            p_model = module.get_parameter(parameter_name)
            p_model.copy_(p)
            p_model.data = p_model.data.to(torch.int8)

        del module.quantised_state

    return dequantise_hook, cleanup_hook

def apply_quantisation_hooks(model,leaf_modules, parameter_mapping):
    
    for leaf_module in leaf_modules:
        dequantise_hook, cleanup_hook = hook_factory(leaf_module, parameter_mapping)
        model.get_submodule(leaf_module).register_forward_pre_hook(dequantise_hook)
        model.get_submodule(leaf_module).register_forward_hook(cleanup_hook)

class Perplexity(evaluate.Metric):
    """An adaptation of HuggingFace's perplexity metric to support custom models."""
    def _info(self):

        @dataclass
        class metric_info():
            inputs_description = ""

        info = metric_info()
        return info

    def _compute(
        self, predictions, model, tokenizer, batch_size: int = 16, add_start_token: bool = True, device=None, max_length=None
    ):

        if device is not None:
            assert device in ["gpu", "cpu", "cuda"], "device should be either gpu or cpu."
            if device == "gpu":
                device = "cuda"
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        model = model.to(device)

        # if batch_size > 1 (which generally leads to padding being required), and
        # if there is not an already assigned pad_token, assign an existing
        # special token to also be the padding token
        if tokenizer.pad_token is None and batch_size > 1:
            existing_special_tokens = list(tokenizer.special_tokens_map_extended.values())
            # check that the model already has at least one special token defined
            assert (
                len(existing_special_tokens) > 0
            ), "If batch_size > 1, model must have at least one special token to use for padding. Please use a different model or set batch_size=1."
            # assign one of the special tokens to also be the pad token
            tokenizer.add_special_tokens({"pad_token": existing_special_tokens[0]})

        if add_start_token and max_length:
            # leave room for <BOS> token to be added:
            assert (
                tokenizer.bos_token is not None
            ), "Input model must already have a BOS token if using add_start_token=True. Please use a different model, or set add_start_token=False"
            max_tokenized_len = max_length - 1
        else:
            max_tokenized_len = max_length

        encodings = tokenizer(
            predictions,
            add_special_tokens=False,
            padding=True,
            truncation=True if max_tokenized_len else False,
            max_length=max_tokenized_len,
            return_tensors="pt",
            return_attention_mask=True,
        ).to(device)

        encoded_texts = encodings["input_ids"]
        attn_masks = encodings["attention_mask"]

        # check that each input is long enough:
        if add_start_token:
            assert torch.all(torch.ge(attn_masks.sum(1), 1)), "Each input text must be at least one token long."
        else:
            assert torch.all(
                torch.ge(attn_masks.sum(1), 2)
            ), "When add_start_token=False, each input text must be at least two tokens long. Run with add_start_token=True if inputting strings of only one token, and remove all empty input strings."

        ppls = []
        loss_fct = CrossEntropyLoss(reduction="none")

        for start_index in logging.tqdm(range(0, len(encoded_texts), batch_size)):
            end_index = min(start_index + batch_size, len(encoded_texts))
            encoded_batch = encoded_texts[start_index:end_index]
            attn_mask = attn_masks[start_index:end_index]

            if add_start_token:
                bos_tokens_tensor = torch.tensor([[tokenizer.bos_token_id]] * encoded_batch.size(dim=0)).to(device)
                encoded_batch = torch.cat([bos_tokens_tensor, encoded_batch], dim=1)
                attn_mask = torch.cat(
                    [torch.ones(bos_tokens_tensor.size(), dtype=torch.int64).to(device), attn_mask], dim=1
                )

            labels = encoded_batch

            with torch.no_grad():
                out_logits = model(encoded_batch, attention_mask=attn_mask).logits

            shift_logits = out_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_attention_mask_batch = attn_mask[..., 1:].contiguous()

            perplexity_batch = torch.exp(
                (loss_fct(shift_logits.transpose(1, 2), shift_labels) * shift_attention_mask_batch).sum(1)
                / shift_attention_mask_batch.sum(1)
            )

            ppls += perplexity_batch.tolist()

        return {"perplexities": ppls, "mean_perplexity": np.mean(ppls)}