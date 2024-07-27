# Jaql
`Jaql`, pronounced "Jekyll", stands for Just Another Quantisation Library. 
It implements similar functionality to `LLama.cpp`, natively supporting quantisation of any PyTorch model, for example most models available on Huggingface. 
Once the model is quantised, `Jaql` adds pre and post forward hooks to manage the inference such that only the layer that is being processed is de-quantised.
When processing a specific layer, the pre-forward hooks de-quantise the layer's weights and recover a `float32` state dict, while the rest of the model remains in `int8`.
The post-forward hooks resert the state dict to the quantised weights before moving on to subsequent layers.
The quantisation and de-quantisation functions use so called scalar quantisation, where a tensor of `float32` is mapped to a specific range which overlaps the range of a lower precision format, in this case `int8`.
The quantisation operation can be viewed as the discretisation of an affine transformation on the input tensor:

```math
t_q = \text{convert}(t * scale + location)
```
where the `convert` converts from a `float32` to `int8`, and the scale/location parameters are computed as:
```math
scale = (t_{max} - t_{min})/(int8_{max} - int8_{min})
```
```math
zero_q = (int8_{max} + int8_{min})/2
```
```math
zero_t = (t_{max} + t_{min})/2
```
```math
location = zero_q - scale * zero_t
```

`Jaql` performs quantisation on chunks of tensor, to improve the performance. 
The chunk size is generally set to `32`, as in `LLama.cpp`.

# Performance
`Jaql` can quantise a `GPT2-small` model (120 million parameters) in less than a second on a `g4dn.xlarge` AWS instance.
The inference speed is slightly slower (20%) with respect to a non-quantised model.
I suspect this is mostly due to how weights are quantised, as PyTorch doesn't easily accept converting parameters of the state dict to `int8`.
This required re-allocating substantially more tensors, which slows down the de-quantisation hooks.

To evaluate the model degradation I follow `LLama.cpp` in computing perplexity change. 
On the `wikitext-2-raw-v1` dataset, the quantised `GPT2-small` model achieves a 9% higher perplexity (worse) than the non-quantised model.
Even though this is computed using the official implementation in HuggingFace `evaluate`, the perplexity values are much higher (~6x) than the ones reported in the GPT2 paper, even for the non-quantised version.
This implies that perplexity was likely measured with a different procedure, likely involving some additional filtering on the dataset, and this might over/under estimate the perplexity degradation.
Some metrics from LLama quantisation types: https://www.reddit.com/r/LocalLLaMA/comments/142q5k5/updated_relative_comparison_of_ggml_quantization/.
Info on the LLama K-quants: https://www.reddit.com/r/LocalLLaMA/comments/1dved4c/llamacpp_kquants/.

# Known issues
The only compatibility issue encountered is relative to models which have tied weights. 
This is a common operation in LMs, where the embedding matrix is used both to embed tokens at the head of the model both to predict tokens at the tail, where the same matrix is used, simply transposed. 
While most of the models have a well structured attribute containing a list of such parameters, some models seem to adhere to a different standard which I didn't further investigate. 
