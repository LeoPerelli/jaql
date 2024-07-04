import torch
import math


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


t = torch.rand((10, 100))

t_q, scales, locations = quantise_tensor(t, chunk_size=64)
t_approx = dequantise_tensor(t_q, scales, locations, 64)
compute_quantisation_mse(t, t_approx)
