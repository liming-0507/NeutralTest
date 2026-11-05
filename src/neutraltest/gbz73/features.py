"""The 45 test features of the FP8 training-operator clause, as data.

The catalogue below is the machine-readable form of the coverage matrix that
accompanies the standard: one row per feature, the operator columns in a fixed
order, and a tier telling the reader whether the standard demands the row
("required"), makes it conditional on the operator's capabilities
("conditional"), or merely describes it ("optional").

The per-operator bits record where the *standard's text* already writes a
requirement. They are kept so that this package can print the same table next
to the results it actually produced -- the gap between "written in the
standard" and "instrumented here" is the interesting part, and it should never
have to be reconstructed by hand.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Operator column order, shared by the coverage bits and the result matrix.
COLUMNS: tuple[str, ...] = ("grouped_gemm", "layer_norm", "gelu", "relu", "amax")

#: Tier of a feature: required by the standard, conditional, or descriptive only.
TIERS: tuple[str, ...] = ("required", "conditional", "optional")

#: Single-letter tier codes used by the source table (m/c/o).
_TIER_FROM_CODE = {"m": "required", "c": "conditional", "o": "optional"}


@dataclass(frozen=True, slots=True)
class Feature:
    """One numbered test feature of the clause."""

    id: str
    title: str
    tier: str
    group: str
    bits: str
    note: str = ""

    def __post_init__(self) -> None:
        if self.tier not in TIERS:
            raise ValueError(f"{self.id}: unknown tier {self.tier!r}")
        if len(self.bits) != len(COLUMNS):
            raise ValueError(
                f"{self.id}: expected {len(COLUMNS)} coverage bits, got {len(self.bits)!r}"
            )

    @classmethod
    def of(cls, fid: str, title: str, code: str, group: str, bits: str, note: str = "") -> Feature:
        return cls(
            id=fid, title=title, tier=_TIER_FROM_CODE[code], group=group, bits=bits, note=note
        )

    def covered_by_standard(self, op: str) -> bool:
        """Whether the standard's text states a requirement for ``op``."""
        return self.bits[COLUMNS.index(op)] == "1"


def _f(fid: str, title: str, code: str, group: str, bits: str, note: str = "") -> Feature:
    return Feature.of(fid, title, code, group, bits, note)


#: The 45 features, in the order the standard lists them.
FEATURES: tuple[Feature, ...] = (
    _f("A1", "参考一致性（与 FP32/框架参考比对）", "m", "A", "11111"),
    _f("A2", "量化往返（quant→dequant 误差单测）", "m", "A", "00000"),
    _f("A3", "上溢 / 饱和处理", "m", "A", "00100"),
    _f("A4", "下溢 / 次正规数", "c", "A", "00000"),
    _f("A5", "舍入规则（RTNE/RZ 等）", "c", "A", "00000"),
    _f("A6", "NaN / Inf 传播", "m", "A", "00011"),
    _f("A7", "±0 与负零", "c", "A", "00010"),
    _f("A8", "累加 / 归约精度", "m", "A", "01000"),
    _f("A9", "混合精度计算路径", "m", "A", "00100"),
    _f("A10", "scale 语义（应用/更新/隔离）", "m", "A", "11001"),
    _f("A11", "训练态误差扩散（跨 step）", "o", "A", "00000"),
    _f("A12", "梯度正确性（反向）", "m", "A", "00000"),
    _f("B1", "large tensor（>2³¹ 元素 / >2 GB）", "c", "B", "00000"),
    _f("B2", "int64 偏移（>2³¹ 下标）", "c", "B", "00000"),
    _f("B3", "0 / 1 元素与退化 shape", "m", "B", "10000"),
    _f("B4", "非规则形状（非 2 幂 / 非对齐）", "m", "B", "10000"),
    _f("B5", "极端维度（超大/超小/高秩）", "c", "B", "11000"),
    _f("B6", "广播 / 形状不一致", "c", "B", "00000"),
    _f("B7", "布局 / stride（非连续）", "m", "B", "00000"),
    _f("B8", "batch 一致性（样本间隔离）", "m", "B", "10000"),
    _f("B9", "输入数据分布", "m", "B", "11111"),
    _f("C1", "dtype 矩阵（E4M3/E5M2/MXFP8）", "m", "C", "00000"),
    _f("C2", "MX 块（32 元素 + E8M0）", "c", "C", "00000"),
    _f("C3", "scale 粒度 / 布局", "m", "C", "10001"),
    _f("C4", "混合 dtype 组合列表", "c", "C", "00000"),
    _f("C5", "输出 dtype", "m", "C", "00000"),
    _f("D1", "框架语义一致（参数/报错契约）", "m", "D", "00000"),
    _f("D2", "参数边界与取值扫描", "m", "D", "10001"),
    _f("D3", "原位 / 别名 / view", "m", "D", "00000"),
    _f("D4", "变长 / 结构特例", "c", "D", "10000"),
    _f("D5", "新旧接口版本", "o", "D", "00000"),
    _f("D6", "错误处理（非法输入）", "m", "D", "00000"),
    _f("E1", "确定性（重复运行位级一致）", "m", "E", "00001"),
    _f("E2", "随机数（seed / 分布）", "c", "E", "00000"),
    _f("E3", "并发与多流", "o", "E", "00000"),
    _f("E4", "内存安全（越界 / 别名）", "m", "E", "00000"),
    _f("E5", "资源上限（显存/寄存器）", "o", "E", "00000"),
    _f("E6", "多卡 / 切卡", "c", "E", "00000"),
    _f("E7", "环境与可重复部署", "o", "E", "00000"),
    _f("F1", "时延 / 吞吐", "m", "F", "11111"),
    _f("F2", "规模扫描曲线", "m", "F", "11011"),
    _f("F3", "基线对比", "m", "F", "11110"),
    _f("F4", "host 侧开销（launch/metadata）", "c", "F", "10000"),
    _f("F5", "测量协议（warmup/迭代/计时）", "m", "F", "11111"),
    _f("F6", "N/A 项与披露", "m", "F", "00000"),
)

BY_ID: dict[str, Feature] = {f.id: f for f in FEATURES}


def features_of(group: str) -> tuple[Feature, ...]:
    """All features of one group (``A``..``F``)."""
    return tuple(f for f in FEATURES if f.group == group)
