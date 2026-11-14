# neutralTest

**面向 PyTorch 算子的中立多后端一致性测试框架（vendor-neutral）。**

`neutraltest` 是一个 pytest 插件 + 库，只回答一个问题：

> 换一块加速卡、换一个 PyTorch 构建，同样的输入、同样的 dtype、前向**和**反向，
> 算出来的数是不是和参考实现一致？

而它不需要知道这块卡是谁家的。

```console
$ neutraltest info
neutraltest 0.1.0
python      3.13.13 (x86_64)
torch       2.15.0+cpu
registered  privateuseone

               mechanism    device_type    available  devices
-------------  -----------  -------------  ---------  -------
  privateuse1  privateuse1  privateuseone  no         0
  xpu          xpu          xpu            no         0
  cuda         cuda         cuda           no         0
  mps          mps          mps            no         0
* cpu          cpu          cpu            yes        1

* selected by --backend=auto
```

## 为什么要这样做

PyTorch 的 dispatch 层只认**固定的少数几种 device type**：`cpu`、`cuda`、
`xpu`、`mps`、`meta`，外加一个留给外部后端的槽位 `privateuse1`
（可通过 `torch.utils.rename_privateuse1_backend` 改名，一个进程只能改一次）。
加速卡厂商拿不到"一人一个 device type"的待遇，只能二选一：

- 抢占那唯一的可改名槽位（`privateuse1`）；
- 或者维护一个给 PyTorch 打补丁、自己新增 device type 的分支。

由此产生两个后果，都伤测试：

1. 按 `device="cuda"` 写的测试套件，会**静默跳过**所有非 CUDA 的加速卡；
2. 每家厂商各写一套测试，容差各定各的、"通过"的定义各不一样，
   跨厂商的测试结果无法比较。

`neutraltest` 反过来做：**测机制，不测品牌**。框架把测试按机器上真实存在的
dispatch 机制（`cpu` / `cuda` / `xpu` / `mps` / `privateuse1`）参数化；
适配器最终指向哪块卡，用的是同一套算子矩阵、同一套容差、同一个参考实现，
因此换机器、换厂商，结果可以直接对比。

本仓库不含任何厂商私有信息：没有厂商名字、没有厂商头文件、没有厂商运行时。
一个后端由"把某机制映射到某 device type"的适配器描述，
适配器也可以作为 entry point 在本仓库之外单独发布（见
[docs/backends.md](docs/backends.md)）。

## 安装

```bash
pip install neutraltest
```

从源码安装：

```bash
python -m venv .venv --system-site-packages   # 复用已装好的 CPU torch
.venv/bin/pip install -e . --no-deps
```

要求 Python ≥ 3.10、`torch` ≥ 2.1、`pytest` ≥ 7.4。

## 快速开始

```bash
neutraltest info                    # 这台机器上有哪些后端
neutraltest doctor                  # 每个可用后端跑 7 项冒烟检查
neutraltest test                    # 跑自带的算子一致性套件
neutraltest test --backend=cpu -k matmul
```

也可以直接驱动 pytest（插件通过 `pytest11` entry point 注册，不需要 conftest）：

```bash
pytest --backend=cpu src/neutraltest/conformance
pytest --backend=all --on-unsupported=skip      # 不可用后端跳过而不是失败
pytest --backend=cpu,xpu --nt-report=report.json
```

### `--backend` 取值

| 写法 | 含义 |
| --- | --- |
| `auto`（默认） | 所有本机可用的后端 |
| `all` | 所有已发现后端（不可用的显示为 skip） |
| `cpu` | 按名字指定 |
| `cpu,xpu` | 逗号列表，保持顺序 |
| `*pu` | 对名字/device type/别名做 glob |

未显式传参时回落到环境变量 `NEUTRALTEST_BACKEND`，方便 CI 切换后端而不改命令行。

### 选项

| 选项 | 默认 | 含义 |
| --- | --- | --- |
| `--backend` | `auto` | 后端选择（见上） |
| `--device-index` | `0` | 用每个后端的第几张卡 |
| `--list-backends` | — | 打印后端表并退出 |
| `--backend-strict` | 关 | 请求的后端不可用时按用法错误处理，而不是 skip |
| `--on-unsupported` | `fail` | 算子未实现时的行为：`fail` / `skip` / `xfail` |
| `--tolerance-multiplier` | `1.0` | 全局容差缩放 |
| `--nt-seed` | `20240101` | 输入生成种子 |
| `--nt-report PATH` | — | 输出 JSON 结果报告 |
| `--nt-no-summary` | 关 | 不打印终端汇总表 |

不想改测试代码就想缩小范围时用标记：`nt_backend`、`nt_exclude_backend`、
`nt_family`、`nt_op`、`nt_grad`、`nt_slow`。

### 跑完长什么样

```console
$ pytest -q --backend=cpu src/neutraltest/conformance
=============================== neutraltest summary ===============================
backend  passed  failed  skipped  xfailed  xpassed  error  unsupported  time
-------  ------  ------  -------  -------  -------  -----  -----------  -----
cpu      161     0       0        0        0        0      0            0.34s
```

`unsupported` 列统计"后端明确说没有这个 kernel"的用例。默认算失败；
`--on-unsupported=skip` 把它们变成 skip，这样只实现了一部分算子的后端
也能看清"哪些是支持的、哪些是缺的"。

## 一条测试是怎么组成的

测试是**数据**而不是代码：`OpCase` 描述算子、输入构造方式、要测的 dtype，
以及它属于哪一类数值行为。

```python
import torch
from neutraltest import OpCase, check_case, randn

def _build(dtype, generator):
    x = randn((8, 16), dtype, generator)
    y = randn((16,), dtype, generator)
    return (x, y), {}

ADD = OpCase(
    id="add/broadcast",
    family="elementwise",
    op="add",
    fn=torch.add,
    build=_build,
    kind="elementwise",
    grad=True,
    grad_inputs=(0, 1),
)

def test_add(nt):
    check_case(ADD, nt.backend, torch.float32)
```

`check_case` 负责全部细节：

1. 用带种子的 generator **在 CPU 上**造输入 —— 设备不需要实现 `randn`，
   且跑设备的一侧与参考实现拿到的是同一批字节；
2. 转成被测 dtype 并搬到设备；
3. 在 CPU 上用 `float64` 重算同一个调用，作为参考值；
4. 先比元信息（返回个数、形状、输出 dtype），**再**比数值，
   这样"dtype 不对"会报成 dtype 问题而不是精度问题；
5. 参考值有限的位置，实际值也必须是有限的；
6. `grad=True` 时打开 `requires_grad` 重放前向，逐输入比较梯度。

因为参考实现就是纯 CPU PyTorch，出现了偏差，责任必然在加速卡——或者在适配器。

## 算子矩阵

自带套件覆盖 **11 个族、81 个用例、151 个 dtype 参数化测试**，
专挑加速卡最容易算错的地方：逐元素、激活、矩阵乘、规约、归一化、
卷积、池化、形状操作、索引/散射、损失、注意力。

完整表格见 [docs/operator-matrix.md](docs/operator-matrix.md)。

## 和庞大的 PyTorch 共处

CPU 版 torch 解开就 800 MB，带 CUDA 栈要几个 GB，所以框架的原则是"复用它，而不是自带一份"：

- 只声明依赖、绝不把 torch 拷进仓库 —— 仓库不到 1 MB，也没有 golden 文件：
  输入按种子现场生成，参考值现场用 CPU float64 算；
- 环境很便宜（`--system-site-packages` 下整个 venv 13 MB），第二份 torch 很贵 ——
  `pip install -e . --no-deps`，别让 pip 再解析一遍 torch；
- CI 统一走 `https://download.pytorch.org/whl/cpu`（约 800 MB，而不是 3–5 GB，
  不带 `nvidia-*`/`triton`/`cudnn`）；
- 兼容性用稀疏矩阵采样 —— 旧 torch 只配一个 Python，其余用最新版，
  由 `tools/torch_matrix.sh` 驱动。

数字与细节见 [docs/space.md](docs/space.md)。

## SDK 版本

PyTorch 之外的另一半是**它下面那整套栈**：加速卡 SDK 是整体发版的——驱动、运行时、算子库、
框架适配层、容器一起重建，绑定在**某一个** PyTorch 版本上。所以「测试挂了」和「测试挂在一个
没人认证过的栈上」是两件不同的事，而后者往往更有用。

`neutraltest sdk` 把这件事量出来，并且只说量得出的：

```
component   version     window        band
----------  ----------  ------------  -----
python      3.13.13     >=3.10,<3.13  above
torch       2.15.0+cpu  >=2.13,<2.14  above
```

- profile 是 JSON（`--sdk examples/sdk-profile.json`），厂商信息不编进代码。
- 偏差是**披露**而不是失败：`--on-sdk-out-of-band warn|skip|fail`，默认 `warn`。
- 进程里读不到版本的组件列为 *not measured*，**不会**被当成匹配。
- 内置的 PyTorch 支持窗口表来自上游 `RELEASE.md` 快照，每次运行都会打印来源与抓取日期。

细节见 [docs/sdk.md](docs/sdk.md)。

## 树外后端怎么接进来

两种集成方式都支持：注册成 `privateuse1`（不用重编 torch），或直接发行一个设备类型
就是 `cuda`/`xpu` 的 torch。重命名只能做一次、`privateuseone` 这个占位名、适配器必须
暴露哪些成员，见 [docs/privateuse1.md](docs/privateuse1.md)；`tests/behavior/test_privateuse1_stub.py`
用桩模块把这条路证了一遍，不需要真卡就能保持它是对的。

## 容差

跨实现比浮点，要的是"预算"，不是常数。`neutraltest` 用 dtype 与数值类别两维决定预算：

| kind | 典型算子 | 系数 |
| --- | --- | --- |
| `exact` | `cat`、`transpose`、索引类 | 0（逐位比较） |
| `elementwise` | `add`、`mul`、`relu`、`max` | 1× |
| `transcendental` | `exp`、`log`、`sin`、`gelu`、`softmax` | 2× |
| `reduction` | `sum`、`mean`、`topk`、`var` | 2× |
| `loss` / `matmul` / `normalization` | `mse_loss`、`matmul`、`layer_norm` | 4× |
| `convolution` / `composite` | `conv2d`、反向传播 | 8× |
| `attention` | `scaled_dot_product_attention` | 16× |

基础预算：`float32` 为 `(rtol=1.3e-6, atol=1e-5)`，`float16` 为 `1e-3`，
`bfloat16` 为 `1.6e-2`，`float64` 为 `1e-12`；整数与布尔一律精确比较。
整数/索引永远走精确路径 —— 差一个下标不是容差问题。
`--tolerance-multiplier` 缩放整张表，这是表达"这块卡确实更松"的诚实方式，
而不是把错了一千倍的算子也放过。

## 接入一个新后端

适配器就是一个小类，说清"怎么够到设备"：

```python
from neutraltest import Backend, Mechanism

class AcceleratorBackend(Backend):
    name = "accelerator"            # --backend 匹配的名字
    mechanism = Mechanism.PRIVATEUSE1

    @property
    def device_type(self) -> str:
        return "vendorx"            # rename_privateuse1_backend() 用的名字

    @property
    def is_available(self) -> bool:
        return torch.vendorx.is_available()
```

打成 entry point 后插件会自动发现：

```toml
[project.entry-points."neutraltest.backends"]
accelerator = "vendorx.torch_test:AcceleratorBackend"
```

完整契约、`privateuse1` 的坑、以及"没有硬件怎么测适配器"：
[docs/backends.md](docs/backends.md)。

## 命令行

| 命令 | 用途 |
| --- | --- |
| `neutraltest info` | 版本、torch 构建、已注册的 `privateuse1` 名、后端表、dtype 支持矩阵、容差预算 |
| `neutraltest doctor` | 每个可用后端跑 7 项冒烟检查（分配、逐元素、矩阵乘、规约、softmax、反向、host↔device 往返），有失败即非零退出 |
| `neutraltest sdk [--sdk FILE]` | 本机环境相对它声称支持的 SDK 发版、以及相对 PyTorch 官方支持窗口的位置 |
| `neutraltest test [pytest 参数]` | 带着后端选项调 pytest，默认跑自带一致性套件 |

`doctor` 是区分"卡不在"和"卡在但算错了"最快的办法。

## 仓库结构

```
src/neutraltest/
  backends/      Backend 抽象、内置适配器、发现与 --backend 解析
  testing.py     OpCase、输入构造、check_case / check_backward
  compare.py     张量比较（精确/容差）、梯度比较
  tolerance.py   dtype × 数值类别 的容差表
  seeding.py     可复现的输入生成
  policy.py      运行策略（未实现处理、容差缩放、种子、卡序号）
  pytest_plugin.py  选项、fixture、标记、汇总、JSON 报告
  report.py      结果记录与汇总表
  cli.py         info / doctor / test
tests/
  unit/          框架自己的测试（不需要加速卡）
  behavior/      用真实 pytest 子进程驱动插件
  conformance/   算子矩阵（随包发布）—— 拿去打设备的就是这部分
tools/         算子矩阵生成器、torch 版本矩阵运行器
docs/
```

## 开发

```bash
python -m venv .venv --system-site-packages
.venv/bin/pip install -e . --no-deps
.venv/bin/python -m pytest tests -q
ruff format --check . && ruff check .
mypy
```

整套测试可以在**完全没有加速卡**的机器上跑：`cpu` 永远存在，
不可用后端显示为 skip，`tests/unit` 根本不碰设备。

## 状态

Alpha。框架与算子矩阵在 CPU PyTorch 上完整跑通；
`cuda` / `xpu` / `mps` / `privateuse1` 适配器在开发机上只做了探测，
因为这台机器没有对应硬件。真卡上的实测结果，正是这个项目最想要社区贡献的东西。

## 许可证

Apache-2.0，见 [LICENSE](LICENSE)。
