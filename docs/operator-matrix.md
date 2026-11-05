# Operator matrix

Every case below is parametrised over its dtypes, so 81 cases become 151 tests.

`kind` selects the tolerance budget (see [writing-tests.md](writing-tests.md#tolerances));
`bwd` marks cases that also compare input gradients against CPU `float64`.


## activation

| case | op | dtypes | kind | bwd | module |
| --- | --- | --- | --- | --- | --- |
| `elu` | `elu` | f32/bf16 | transcendental |  | `neutraltest/conformance/test_activation.py` |
| `gelu` | `gelu` | f32/bf16 | transcendental |  | `neutraltest/conformance/test_activation.py` |
| `glu/dim-1` | `glu` | f32/bf16 | elementwise |  | `neutraltest/conformance/test_activation.py` |
| `hardswish` | `hardswish` | f32/bf16 | elementwise |  | `neutraltest/conformance/test_activation.py` |
| `leaky-relu` | `leaky_relu` | f32/bf16 | elementwise |  | `neutraltest/conformance/test_activation.py` |
| `prelu` | `prelu` | f32/bf16 | elementwise |  | `neutraltest/conformance/test_activation.py` |
| `relu` | `relu` | f32/bf16 | elementwise | yes | `neutraltest/conformance/test_activation.py` |
| `sigmoid` | `sigmoid` | f32/bf16 | transcendental |  | `neutraltest/conformance/test_activation.py` |
| `silu` | `silu` | f32/bf16 | transcendental |  | `neutraltest/conformance/test_activation.py` |
| `softplus` | `softplus` | f32/bf16 | transcendental |  | `neutraltest/conformance/test_activation.py` |

## attention

| case | op | dtypes | kind | bwd | module |
| --- | --- | --- | --- | --- | --- |
| `sdpa/causal` | `scaled_dot_product_attention` | f32 | attention |  | `neutraltest/conformance/test_attention.py` |
| `sdpa/full` | `scaled_dot_product_attention` | f32 | attention | yes | `neutraltest/conformance/test_attention.py` |

## convolution

| case | op | dtypes | kind | bwd | module |
| --- | --- | --- | --- | --- | --- |
| `conv-transpose2d` | `conv_transpose2d` | f32 | convolution |  | `neutraltest/conformance/test_convolution.py` |
| `conv1d/pad1` | `conv1d` | f32/bf16 | convolution |  | `neutraltest/conformance/test_convolution.py` |
| `conv2d/groups-2` | `conv2d` | f32 | convolution |  | `neutraltest/conformance/test_convolution.py` |
| `conv2d/pad1` | `conv2d` | f32/bf16 | convolution | yes | `neutraltest/conformance/test_convolution.py` |
| `conv2d/stride-2` | `conv2d` | f32/bf16 | convolution |  | `neutraltest/conformance/test_convolution.py` |

## elementwise

| case | op | dtypes | kind | bwd | module |
| --- | --- | --- | --- | --- | --- |
| `abs` | `abs` | f32/bf16 | elementwise |  | `neutraltest/conformance/test_elementwise.py` |
| `add/broadcast` | `add` | f32/bf16 | elementwise | yes | `neutraltest/conformance/test_elementwise.py` |
| `clamp` | `clamp` | f32/bf16 | elementwise |  | `neutraltest/conformance/test_elementwise.py` |
| `cos` | `cos` | f32/bf16 | transcendental |  | `neutraltest/conformance/test_elementwise.py` |
| `div/broadcast` | `div` | f32/bf16 | elementwise |  | `neutraltest/conformance/test_elementwise.py` |
| `exp` | `exp` | f32/bf16 | transcendental |  | `neutraltest/conformance/test_elementwise.py` |
| `log` | `log` | f32/bf16 | transcendental |  | `neutraltest/conformance/test_elementwise.py` |
| `maximum/broadcast` | `maximum` | f32/bf16 | elementwise |  | `neutraltest/conformance/test_elementwise.py` |
| `mul/broadcast` | `mul` | f32/bf16 | elementwise |  | `neutraltest/conformance/test_elementwise.py` |
| `neg` | `neg` | f32/bf16 | elementwise |  | `neutraltest/conformance/test_elementwise.py` |
| `pow/scalar` | `pow` | f32/bf16 | transcendental |  | `neutraltest/conformance/test_elementwise.py` |
| `sin` | `sin` | f32/bf16 | transcendental |  | `neutraltest/conformance/test_elementwise.py` |
| `sqrt` | `sqrt` | f32/bf16 | transcendental |  | `neutraltest/conformance/test_elementwise.py` |
| `sub/broadcast` | `sub` | f32/bf16 | elementwise |  | `neutraltest/conformance/test_elementwise.py` |
| `tanh` | `tanh` | f32/bf16 | transcendental |  | `neutraltest/conformance/test_elementwise.py` |
| `where/broadcast` | `where` | f32/bf16 | elementwise |  | `neutraltest/conformance/test_elementwise.py` |

## indexing

| case | op | dtypes | kind | bwd | module |
| --- | --- | --- | --- | --- | --- |
| `embedding` | `embedding` | f32/bf16 | exact |  | `neutraltest/conformance/test_indexing.py` |
| `gather/dim-1` | `gather` | f32/bf16 | exact |  | `neutraltest/conformance/test_indexing.py` |
| `index-select` | `index_select` | f32/bf16 | exact |  | `neutraltest/conformance/test_indexing.py` |
| `masked-fill` | `masked_fill` | f32/bf16 | exact |  | `neutraltest/conformance/test_indexing.py` |
| `scatter/dim-1` | `scatter` | f32/bf16 | exact |  | `neutraltest/conformance/test_indexing.py` |

## loss

| case | op | dtypes | kind | bwd | module |
| --- | --- | --- | --- | --- | --- |
| `bce-with-logits` | `binary_cross_entropy_with_logits` | f32/bf16 | loss |  | `neutraltest/conformance/test_loss.py` |
| `cross-entropy` | `cross_entropy` | f32 | loss |  | `neutraltest/conformance/test_loss.py` |
| `kl-div` | `kl_div` | f32 | loss |  | `neutraltest/conformance/test_loss.py` |
| `l1-loss` | `l1_loss` | f32/bf16 | loss |  | `neutraltest/conformance/test_loss.py` |
| `mse-loss` | `mse_loss` | f32/bf16 | loss | yes | `neutraltest/conformance/test_loss.py` |
| `nll-loss` | `nll_loss` | f32 | loss |  | `neutraltest/conformance/test_loss.py` |
| `smooth-l1-loss` | `smooth_l1_loss` | f32/bf16 | loss |  | `neutraltest/conformance/test_loss.py` |

## matmul

| case | op | dtypes | kind | bwd | module |
| --- | --- | --- | --- | --- | --- |
| `bmm` | `bmm` | f32/bf16 | matmul |  | `neutraltest/conformance/test_matmul.py` |
| `linear` | `linear` | f32/bf16 | matmul | yes | `neutraltest/conformance/test_matmul.py` |
| `matmul/2d` | `matmul` | f32/bf16 | matmul | yes | `neutraltest/conformance/test_matmul.py` |
| `matmul/broadcast-batch` | `matmul` | f32/bf16 | matmul |  | `neutraltest/conformance/test_matmul.py` |

## normalization

| case | op | dtypes | kind | bwd | module |
| --- | --- | --- | --- | --- | --- |
| `batch-norm/training` | `batch_norm` | f32 | normalization |  | `neutraltest/conformance/test_normalization.py` |
| `group-norm` | `group_norm` | f32 | normalization |  | `neutraltest/conformance/test_normalization.py` |
| `layer-norm` | `layer_norm` | f32/bf16 | normalization | yes | `neutraltest/conformance/test_normalization.py` |
| `log-softmax/dim-1` | `log_softmax` | f32/bf16 | normalization |  | `neutraltest/conformance/test_normalization.py` |
| `normalize/l2` | `normalize` | f32/bf16 | normalization |  | `neutraltest/conformance/test_normalization.py` |
| `rms-norm` | `rms_norm` | f32 | normalization |  | `neutraltest/conformance/test_normalization.py` |
| `softmax/dim-1` | `softmax` | f32/bf16 | normalization | yes | `neutraltest/conformance/test_normalization.py` |

## pooling

| case | op | dtypes | kind | bwd | module |
| --- | --- | --- | --- | --- | --- |
| `adaptive-avg-pool2d` | `adaptive_avg_pool2d` | f32/bf16 | reduction |  | `neutraltest/conformance/test_pooling.py` |
| `avg-pool2d/k2` | `avg_pool2d` | f32/bf16 | reduction |  | `neutraltest/conformance/test_pooling.py` |
| `max-pool2d/indices` | `max_pool2d` | f32 | elementwise |  | `neutraltest/conformance/test_pooling.py` |
| `max-pool2d/k2` | `max_pool2d` | f32/bf16 | elementwise |  | `neutraltest/conformance/test_pooling.py` |
| `max-pool2d/k3-s2-p1` | `max_pool2d` | f32/bf16 | elementwise |  | `neutraltest/conformance/test_pooling.py` |

## reduction

| case | op | dtypes | kind | bwd | module |
| --- | --- | --- | --- | --- | --- |
| `amax/dim` | `amax` | f32/bf16 | reduction |  | `neutraltest/conformance/test_reduction.py` |
| `argmax/dim` | `argmax` | f32/bf16 | elementwise |  | `neutraltest/conformance/test_reduction.py` |
| `cumsum/dim` | `cumsum` | f32/bf16 | reduction |  | `neutraltest/conformance/test_reduction.py` |
| `l2-norm/dim` | `vector_norm` | f32/bf16 | reduction |  | `neutraltest/conformance/test_reduction.py` |
| `logsumexp/dim` | `logsumexp` | f32/bf16 | reduction |  | `neutraltest/conformance/test_reduction.py` |
| `max/dim` | `max` | f32/bf16 | elementwise |  | `neutraltest/conformance/test_reduction.py` |
| `mean/dim` | `mean` | f32/bf16 | reduction |  | `neutraltest/conformance/test_reduction.py` |
| `min/dim` | `min` | f32/bf16 | elementwise |  | `neutraltest/conformance/test_reduction.py` |
| `sum/dim` | `sum` | f32/bf16 | reduction | yes | `neutraltest/conformance/test_reduction.py` |
| `topk` | `topk` | f32/bf16 | elementwise |  | `neutraltest/conformance/test_reduction.py` |
| `var/dim` | `var` | f32/bf16 | reduction |  | `neutraltest/conformance/test_reduction.py` |

## shape

| case | op | dtypes | kind | bwd | module |
| --- | --- | --- | --- | --- | --- |
| `cat/dim-1` | `cat` | f32/bf16 | exact |  | `neutraltest/conformance/test_manipulation.py` |
| `flip/dim-1` | `flip` | f32/bf16 | exact |  | `neutraltest/conformance/test_manipulation.py` |
| `pad/constant` | `pad` | f32/bf16 | exact |  | `neutraltest/conformance/test_manipulation.py` |
| `permute` | `permute` | f32/bf16 | exact |  | `neutraltest/conformance/test_manipulation.py` |
| `reshape` | `reshape` | f32/bf16 | exact |  | `neutraltest/conformance/test_manipulation.py` |
| `roll/dim-1` | `roll` | f32/bf16 | exact |  | `neutraltest/conformance/test_manipulation.py` |
| `split/chunks` | `split` | f32/bf16 | exact |  | `neutraltest/conformance/test_manipulation.py` |
| `stack/dim-0` | `stack` | f32/bf16 | exact |  | `neutraltest/conformance/test_manipulation.py` |
| `transpose` | `transpose` | f32/bf16 | exact |  | `neutraltest/conformance/test_manipulation.py` |

**Total: 81 cases**, 10 of which check gradients.
