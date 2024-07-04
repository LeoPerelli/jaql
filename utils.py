import torch
import math
from tqdm import tqdm

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
    t_q = torch.zeros_like(t_flat).type(torch.int8)
    n_chunks = math.ceil(len(t_flat) / chunk_size)
    scales = torch.zeros(n_chunks)
    locations = torch.zeros(n_chunks)

    for chunk_id in range(n_chunks):

        left = chunk_id * chunk_size
        right = min(len(t_flat), (chunk_id + 1) * chunk_size)

        t_q[left:right], scales[chunk_id], locations[chunk_id] = scalar_quantisation(t_flat[left:right])

    t_q = t_q.reshape(shape)

    return t_q, scales, locations


def dequantise_tensor(t_q, scales, locations, chunk_size):

    shape = t_q.shape
    t_q = t_q.flatten()

    n_chunks = len(scales)
    t = torch.zeros_like(t_q).type(torch.float32)

    for chunk_id in range(n_chunks):
        left = chunk_id * chunk_size
        right = min(len(t_q), (chunk_id + 1) * chunk_size)

        t[left:right] = scalar_dequantisation(t_q[left:right], scales[chunk_id], locations[chunk_id])

    t = t.reshape(shape)
    return t

def quantise_model(model, chunk_size):

    parameter_mapping = {}
    for parameter_name, p in tqdm(list(model.named_parameters()), desc="Quantising model layers"):
        p_q, scales, locations = quantise_tensor(p.clone(), chunk_size)
        p.data.copy_(p_q)
        p.requires_grad = False
        p.data = p.data.to(torch.int8)
        parameter_mapping[parameter_name] = {'scales':scales, 'locations': locations, 'chunk_size': chunk_size}

    leaf_modules = get_leaf_modules(model)
    apply_quantisation_hooks(model, leaf_modules, parameter_mapping)

    return model, parameter_mapping

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

    return dequantise_hook, cleanup_hook

def apply_quantisation_hooks(model,leaf_modules, parameter_mapping):
    
    for leaf_module in leaf_modules:
        dequantise_hook, cleanup_hook = hook_factory(leaf_module, parameter_mapping)
        model.get_submodule(leaf_module).register_forward_pre_hook(dequantise_hook)
        model.get_submodule(leaf_module).register_forward_hook(cleanup_hook)