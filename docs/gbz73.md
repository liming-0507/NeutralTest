# The FP8 clause 7.3 matrix

The accelerator interface standard for FP8 inference has a clause 7.3 whose
table lists 45 test items for five operator families: grouped GEMM,
LayerNorm, GELU, ReLU and Amax. This package is that table, executable.

Each row is one script. `items/a7.py` is the ±0 row and nothing else, so a
reviewer holding the standard can open the clause, find the row, and read the
code that tests it. The table below is generated from `features.py`, which is
generated from the standard's own text — the row titles and the tier of each
row are the standard's wording, not a paraphrase.

## The two baselines

Every feature is judged against two references, both computed on the CPU:

| baseline | dtype | role |
| --- | --- | --- |
| ordinary standard | `float32` | the result a normal framework gives |
| golden standard | `float64` | the result the arithmetic actually has |

`ctx.normal()` returns the first, `ctx.golden()` returns the second. A feature
that compares against the golden one is asking "is this right?"; a feature that
compares against the ordinary one is asking "is this as good as the float
path?" — both questions appear in the clause, and conflating them is how a
suite ends up either too strict or too forgiving.

## The five operators

| operator | family | what it stands for |
| --- | --- | --- |
| `grouped_gemm` | gemm | grouped matrix multiply with a cumulative group list |
| `layer_norm` | normalisation | normalisation with weight and bias |
| `gelu` | activation | the smooth activation, exact and tanh approximations |
| `relu` | activation | the simplest elementwise kernel |
| `amax` | reduction | the reduction that produces an FP8 scale |

They are ordinary PyTorch expressions, not kernels: the suite's subject is the
*contract*, so a backend that implements the operator however it likes can be
held to the same rows. `ops.py` holds both the reference expression and the
input builder for each of them.

## Tolerances, and where they come from

A tolerance that is written down as a constant is a guess. The floor used here
is derived from the format:

```
floor = ROUNDING_EVENTS x half_ulp(dtype) x family amplification
```

`ROUNDING_EVENTS` is 2 — inputs rounded once, the output rounded once.
`half_ulp(float8_e4m3fn)` is 0.0625, half a step of a 3-bit mantissa.
Amplification is 1.0 for elementwise work, 2.0 for normalisation and GEMM (the
scale is estimated from the data, so the estimate can be off by a step), and
1.0 for reductions.

| family | floor at e4m3 |
| --- | --- |
| activation | 1.25e-1 |
| reduction | 1.25e-1 |
| normalisation | 2.5e-1 |
| gemm | 2.5e-1 |

The measured error of the reference implementation sits at 15–37 % of these
floors, which is the margin the derivation predicted. Two features deliberately
do *not* use the aggregate floor: A2 and A5 check the format itself, where the
correct instrument is the per-element half-ULP bound against the exponent
bucket (`within_half_ulp` in `items/_support.py`) — a relative bound is
meaningless next to zero, where the distance to the neighbour shrinks with the
exponent.

## The catalogue

45 rows: 27 required, 13 conditional, 5 optional.

| row | clause item | tier | script |
| --- | --- | --- | --- |
| A1 | 参考一致性（与 FP32/框架参考比对） | required | `items/a1.py` |
| A2 | 量化往返（quant→dequant 误差单测） | required | `items/a2.py` |
| A3 | 上溢 / 饱和处理 | required | `items/a3.py` |
| A4 | 下溢 / 次正规数 | conditional | `items/a4.py` |
| A5 | 舍入规则（RTNE/RZ 等） | conditional | `items/a5.py` |
| A6 | NaN / Inf 传播 | required | `items/a6.py` |
| A7 | ±0 与负零 | conditional | `items/a7.py` |
| A8 | 累加 / 归约精度 | required | `items/a8.py` |
| A9 | 混合精度计算路径 | required | `items/a9.py` |
| A10 | scale 语义（应用/更新/隔离） | required | `items/a10.py` |
| A11 | 训练态误差扩散（跨 step） | optional | `items/a11.py` |
| A12 | 梯度正确性（反向） | required | `items/a12.py` |
| B1 | large tensor（>2³¹ 元素 / >2 GB） | conditional | `items/b1.py` |
| B2 | int64 偏移（>2³¹ 下标） | conditional | `items/b2.py` |
| B3 | 0 / 1 元素与退化 shape | required | `items/b3.py` |
| B4 | 非规则形状（非 2 幂 / 非对齐） | required | `items/b4.py` |
| B5 | 极端维度（超大/超小/高秩） | conditional | `items/b5.py` |
| B6 | 广播 / 形状不一致 | conditional | `items/b6.py` |
| B7 | 布局 / stride（非连续） | required | `items/b7.py` |
| B8 | batch 一致性（样本间隔离） | required | `items/b8.py` |
| B9 | 输入数据分布 | required | `items/b9.py` |
| C1 | dtype 矩阵（E4M3/E5M2/MXFP8） | required | `items/c1.py` |
| C2 | MX 块（32 元素 + E8M0） | conditional | `items/c2.py` |
| C3 | scale 粒度 / 布局 | required | `items/c3.py` |
| C4 | 混合 dtype 组合列表 | conditional | `items/c4.py` |
| C5 | 输出 dtype | required | `items/c5.py` |
| D1 | 框架语义一致（参数/报错契约） | required | `items/d1.py` |
| D2 | 参数边界与取值扫描 | required | `items/d2.py` |
| D3 | 原位 / 别名 / view | required | `items/d3.py` |
| D4 | 变长 / 结构特例 | conditional | `items/d4.py` |
| D5 | 新旧接口版本 | optional | `items/d5.py` |
| D6 | 错误处理（非法输入） | required | `items/d6.py` |
| E1 | 确定性（重复运行位级一致） | required | `items/e1.py` |
| E2 | 随机数（seed / 分布） | conditional | `items/e2.py` |
| E3 | 并发与多流 | optional | `items/e3.py` |
| E4 | 内存安全（越界 / 别名） | required | `items/e4.py` |
| E5 | 资源上限（显存/寄存器） | optional | `items/e5.py` |
| E6 | 多卡 / 切卡 | conditional | `items/e6.py` |
| E7 | 环境与可重复部署 | optional | `items/e7.py` |
| F1 | 时延 / 吞吐 | required | `items/f1.py` |
| F2 | 规模扫描曲线 | required | `items/f2.py` |
| F3 | 基线对比 | required | `items/f3.py` |
| F4 | host 侧开销（launch/metadata） | conditional | `items/f4.py` |
| F5 | 测量协议（warmup/迭代/计时） | required | `items/f5.py` |
| F6 | N/A 项与披露 | required | `items/f6.py` |

## Running it

```bash
neutraltest gbz73 --backend=cpu          # the whole matrix, as a table
neutraltest gbz73 --backend=cpu --json   # the same, as JSON
neutraltest gbz73 --ops gelu --features A1,A2,A5
pytest src/neutraltest/conformance/test_gbz73.py -q          # 225 pytest cases
pytest src/neutraltest/conformance/test_gbz73.py -q -k "A2-"  # one row, all operators
```

## Reading the result

```
id   tier         feature                            text   groupe layer_   gelu   relu   amax
A9   required     混合精度计算路径                           00100       P      P      -      -      -
```

`P` passed, `F` failed, `-` not applicable. A `-` is never silent: every one is
printed again by `render_disclosure` with its reason, and F6 fails the build if
a row is not applicable without one. "Not applicable" therefore means "this
operator has no such surface", never "this operator would have failed".

The three structural `-` groups are:

- A9/B6 — the operator is unary, so there is no second operand to arrange or
  broadcast.
- B2/D4 — the operator has no integer offset and no variable-length structure.
- E3/E5/E6 — the backend exposes no streams, no register accounting, and a
  single device; these are properties of the machine, so they are reported per
  backend rather than assumed.

## Adding a row

The clause is the specification, so adding a row means adding the script the
clause describes:

```python
# items/g1.py
ID = "G1"        # must match the file name and features.py
OPS = None       # or ("gelu",) to restrict the row to some operators

def run(ctx):
    ...
```

`run` may raise `AssertionError` with an explanation, call `ctx.na(reason)` when
the row genuinely does not apply, and use `ctx.note(...)` to record evidence
that shows up in the JSON report. Then run the matrix: an unimplemented row
fails F6, which is the completeness check.

## Provenance

The operator list, the row titles and their tiers come from the published
standard. The five operator implementations are plain PyTorch expressions
written for this suite. No vendor code, name or path appears here: the suite is
about the contract, and it has to be runnable by anyone with a CPU.
