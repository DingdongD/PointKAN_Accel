"""
PointKAN-Elite Network Definition
===================================

Manual definition of PointKAN-Elite network layers for CIM hardware mapping.
Defines the complete forward-pass layer sequence with per-layer operator lists.

Architecture overview (PointKAN-Elite classification, ModelNet40):
    Input [B, 3, 1024]
     → Embedding: Conv1d(3→32) + BN + ReLU
     → Stage 0~3 (x4, each: points halved, channels doubled):
         ├ GAM: FPS → KNN → AffineNorm → S-Pool       (same as PointKAN, inference fixed)
         ├ LFP: transfer1 → LayerNorm → Efficient_KAN → Add(residual) → AdaptiveMaxPool1d(1);
         │      transfer2 → LayerNorm → Efficient_KAN → Add(residual); Add(fusion)
         └ GFP: Conv1d→BN→ReLU → Conv1d→BN → Add(residual) → ReLU  (same as PointKAN)
     → Global MaxPool
     → Classifier: Linear→BN1d→ReLU (x2) → Linear

Key differences from standard PointKAN:
  1. LFP uses Pre-Norm residual: x = x + Efficient_KAN(LayerNorm(x))
     (vs PointKAN Post-Norm: LayerNorm(KANLayer(x)))
  2. LFP uses Efficient_KAN (KAT_Group rational activation + Linear)
     instead of KANLayer (B-spline KAN + DWConv)
  3. No DWConv inside KAN

Efficient_KAN 与 KAT 统计约定（与 PointKAN-elite 一致）:
  - 真实前向: x → act1(KAT_Group, mode="identity") → fc1 → act2(KAT_Group, mode=gelu) → fc2
  - act1 为 Identity（init.json 中 numerator=[0,1,0,0,0,0]），不占 cycle/energy。
  - 仿真仅对「有效」KAT 建一层算子: 每个 Efficient_KAN 只发一个 KAN 事件，对应 act2；
    act1 不建算子、不参与 cycle/energy 汇总，故统计的 KAN 即为有效 KAT_GROUP 部分。

"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

from .pointkan_net import (
    OperatorRef,
    LayerDef,
    _build_gam_operators,
    _build_gfp_operators,
)


# ---------------------------------------------------------------------------
# PointKAN-Elite configuration
# ---------------------------------------------------------------------------

@dataclass
class PointKANEliteConfig:
    """
    PointKAN-Elite network configuration.

    Attributes:
        name: Network name
        task: Task type
        input_points: Number of input points
        num_classes: Number of output classes
        embed_dim: Initial embedding dimension
        num_stages: Number of feature extraction stages
        dim_expansion: Channel expansion factor per stage
        reducers: Point reduction factor per stage
        k_neighbors: Number of KNN neighbors per stage
    """
    name: str = "PointKAN-Elite"
    task: str = "classification"
    input_points: int = 1024
    num_classes: int = 40
    embed_dim: int = 32
    num_stages: int = 4
    dim_expansion: List[int] = field(default_factory=lambda: [2, 2, 2, 2])
    reducers: List[int] = field(default_factory=lambda: [2, 2, 2, 2])
    k_neighbors: List[int] = field(default_factory=lambda: [24, 24, 24, 24])
    coord_dims: int = 3


def _canonical_classification_stage_points(
    input_points: int,
    reducers: List[int],
    *,
    base_input_points: int = 1024,
) -> List[int]:
    """
    Build PointKAN-Elite classification stage point counts.

    Sweep experiments vary the raw input cloud size presented to embedding and
    the first GAM stage, but later stages should keep the canonical classifier
    hierarchy derived from the 1024-point model: [512, 256, 128, 64].
    """
    current = max(1, int(base_input_points))
    stage_points: List[int] = []
    for idx, reduce in enumerate(reducers):
        if idx == 0:
            current = max(1, current // max(1, int(reduce)))
        else:
            current = max(1, current // max(1, int(reduce)))
        stage_points.append(current)
    if stage_points:
        stage_points[0] = max(1, base_input_points // max(1, int(reducers[0])))
    return stage_points


def _canonical_partseg_stage_points(
    input_points: int,
    reducers: List[int],
    *,
    base_input_points: int = 2048,
) -> List[int]:
    """
    Build PointKAN-Elite part-segmentation stage point counts.

    Sweep experiments may vary the raw encoder input cloud size, but the
    canonical part-seg hierarchy should remain aligned with the 2048-point
    model: [512, 128, 32, 8] for reducers [4,4,4,4].
    """
    current = max(1, int(base_input_points))
    stage_points: List[int] = []
    for reduce in reducers:
        current = max(1, current // max(1, int(reduce)))
        stage_points.append(current)
    if stage_points:
        stage_points[0] = max(1, base_input_points // max(1, int(reducers[0])))
    return stage_points


# ---------------------------------------------------------------------------
# LFP operator builder (Elite-specific: Pre-Norm)
# ---------------------------------------------------------------------------

def _build_lfp_operators_elite(
    channels: int,
    num_points: int,
    lfp_in_channels: Optional[int] = None,
    k_neighbors: int = 1,
    num_groups: int = 8,
) -> List[OperatorRef]:
    """
    Build operator list for a PointKAN-Elite LFP layer.

    Pre-Norm residual: x = x + Efficient_KAN(LayerNorm(x))
    Efficient_KAN 展开为显式 linear：fc1(in→hidden)、fc2(hidden→out)，与真实模型
    LFP_blocks_list.*.operation1.0.fc1/fc2、operation2.0.fc1/fc2 严格对齐，便于 MNSIM/仿真 36 层一致。

    PointKAN-Elite LFP structure (per stage):
      Local path:  transfer1 → LayerNorm → [fc1, fc2] (operation1) → Add(residual) → AdaptiveMaxPool1d(1)
      Center path: transfer2 → LayerNorm → [fc1, fc2] (operation2) → Add(residual)
      Fusion:      local + center (element-wise add)

    Args:
        channels: Output channel dimension (= out_channels of this LFP stage).
        num_points: Number of output points.
        k_neighbors: Number of local neighbors in the LFP local KAT path.
        num_groups: KAT_Group group count. Groups split the channel dimension
            and share one GRF/PWL table per group.
        lfp_in_channels: Actual input channels to transfer1/transfer2 Conv1d.
            - use_xyz=False: 2 * last_channel  (= channels when dim_expansion=2)
            - use_xyz=True:  2 * last_channel + 3
            When None, falls back to ``channels`` (backward-compatible default
            for use_xyz=False with dim_expansion=2).
    """
    if lfp_in_channels is None:
        lfp_in_channels = channels
    k_neighbors = max(1, int(k_neighbors))
    num_groups = max(1, int(num_groups))
    local_tokens = max(1, int(num_points) * k_neighbors)
    center_tokens = max(1, int(num_points))
    ops = []

    # --- Local path ---
    # transfer1: Conv1d + BN + ReLU
    ops.append(OperatorRef(
        op_type="conv1d",
        params={"in_channels": lfp_in_channels, "out_channels": channels, "kernel_size": 1, "num_points": num_points},
        description=f"LFP transfer1: Conv1d({lfp_in_channels}→{channels}, k=1)",
    ))
    ops.append(OperatorRef(
        op_type="bn",
        params={"num_features": channels},
        description=f"LFP transfer1: BN({channels})",
    ))
    ops.append(OperatorRef(
        op_type="relu",
        params={"num_features": channels},
        description="LFP transfer1: ReLU",
    ))
    # Pre-Norm: LayerNorm BEFORE Efficient_KAN
    ops.append(OperatorRef(
        op_type="layernorm",
        params={"normalized_shape": channels, "num_points": num_points},
        description=f"LFP local LayerNorm({channels}) [Pre-Norm]",
    ))
    # Efficient_KAN 展开：operation1.fc1 → KAT(GRF) → operation1.fc2（与 pointkanElite LFP operation1 一致）
    # 仅对有效 KAT(act2) 建一层 KAN；act1(Identity) 不建算子、不计 cycle/energy
    ops.append(OperatorRef(
        op_type="linear",
        params={"in_features": channels, "out_features": channels, "tokens": local_tokens},
        description=f"LFP local operation1.fc1 Linear({channels}→{channels})",
    ))
    ops.append(OperatorRef(
        op_type="kan",
        params={
            "in_features": channels,
            "out_features": channels,
            "num_points": num_points,
            "sequence_len": local_tokens,
            "k_neighbors": k_neighbors,
            "num_groups": num_groups,
            "kat_path": "local",
            "variant": "elite",
        },
        description=f"LFP local operation1 KAT/GRF ({channels}→{channels}, rational activation) [act2 only]",
    ))
    ops.append(OperatorRef(
        op_type="linear",
        params={"in_features": channels, "out_features": channels, "tokens": local_tokens},
        description=f"LFP local operation1.fc2 Linear({channels}→{channels})",
    ))
    # Pre-Norm residual add: x + operation1(LayerNorm(x))
    ops.append(OperatorRef(
        op_type="add",
        params={"residual": True, "num_features": channels, "num_points": num_points},
        description="LFP local residual add (Pre-Norm)",
    ))
    # AdaptiveMaxPool1d(output_size=1) on local path
    ops.append(OperatorRef(
        op_type="adaptive_max_pool1d",
        params={"output_size": 1, "num_features": channels, "num_points": num_points},
        description="LFP local AdaptiveMaxPool1d(1)",
    ))

    # --- Center path ---
    # transfer2: Conv1d + BN + ReLU
    ops.append(OperatorRef(
        op_type="conv1d",
        params={"in_channels": lfp_in_channels, "out_channels": channels, "kernel_size": 1, "num_points": num_points},
        description=f"LFP transfer2: Conv1d({lfp_in_channels}→{channels}, k=1)",
    ))
    ops.append(OperatorRef(
        op_type="bn",
        params={"num_features": channels},
        description=f"LFP transfer2: BN({channels})",
    ))
    ops.append(OperatorRef(
        op_type="relu",
        params={"num_features": channels},
        description="LFP transfer2: ReLU",
    ))
    # Pre-Norm: LayerNorm BEFORE Efficient_KAN
    ops.append(OperatorRef(
        op_type="layernorm",
        params={"normalized_shape": channels, "num_points": num_points},
        description=f"LFP center LayerNorm({channels}) [Pre-Norm]",
    ))
    # Efficient_KAN 展开：operation2.fc1 → KAT(GRF) → operation2.fc2（与 pointkanElite LFP operation2 一致）
    # 仅对有效 KAT(act2) 建一层 KAN；act1(Identity) 不建算子、不计 cycle/energy
    ops.append(OperatorRef(
        op_type="linear",
        params={"in_features": channels, "out_features": channels, "tokens": center_tokens},
        description=f"LFP center operation2.fc1 Linear({channels}→{channels})",
    ))
    ops.append(OperatorRef(
        op_type="kan",
        params={
            "in_features": channels,
            "out_features": channels,
            "num_points": num_points,
            "sequence_len": center_tokens,
            "num_groups": num_groups,
            "kat_path": "center",
            "variant": "elite",
        },
        description=f"LFP center operation2 KAT/GRF ({channels}→{channels}, rational activation) [act2 only]",
    ))
    ops.append(OperatorRef(
        op_type="linear",
        params={"in_features": channels, "out_features": channels, "tokens": center_tokens},
        description=f"LFP center operation2.fc2 Linear({channels}→{channels})",
    ))
    # Pre-Norm residual add: center + operation2(LayerNorm(center))
    ops.append(OperatorRef(
        op_type="add",
        params={"residual": True, "num_features": channels, "num_points": num_points},
        description="LFP center residual add (Pre-Norm)",
    ))
    # Fusion: element-wise add (local + center)
    ops.append(OperatorRef(
        op_type="add",
        params={"num_features": channels, "num_points": num_points},
        description="LFP fusion: local + center",
    ))

    return ops


# ---------------------------------------------------------------------------
# Network creation function
# ---------------------------------------------------------------------------

def create_pointkan_elite_classification(
    input_points: int = 1024,
    num_classes: int = 40,
    embed_dim: int = 32,
    dim_expansion: Optional[List[int]] = None,
    reducers: Optional[List[int]] = None,
    k_neighbors: Optional[List[int]] = None,
) -> List[LayerDef]:
    """
    Create PointKAN-Elite classification network layer sequence.

    Default configuration (pointKANElite factory function):
        input: [B, 3, 1024], 40 classes (ModelNet40)
        embed_dim=32, 4 stages, dim_expansion=[2,2,2,2],
        reducers=[2,2,2,2], k_neighbors=[24,24,24,24]

    Args:
        input_points: Number of input points
        num_classes: Number of output classes
        embed_dim: Initial embedding dimension
        dim_expansion: Per-stage channel expansion factors
        reducers: Per-stage point reduction factors
        k_neighbors: Per-stage KNN neighbor counts

    Returns:
        Ordered list of LayerDef defining the complete forward pass.
    """
    if dim_expansion is None:
        dim_expansion = [2, 2, 2, 2]
    if reducers is None:
        reducers = [2, 2, 2, 2]
    if k_neighbors is None:
        k_neighbors = [24, 24, 24, 24]

    num_stages = len(dim_expansion)
    layers: List[LayerDef] = []

    # ------------------------------------------------------------------
    # 1. Embedding: Conv1d(3→embed_dim) + BN + ReLU
    #    (identical to standard PointKAN)
    # ------------------------------------------------------------------
    layers.append(LayerDef(
        name="embedding",
        layer_type="embedding",
        operators=[
            OperatorRef(
                op_type="conv1d",
                params={"in_channels": 3, "out_channels": embed_dim,
                        "kernel_size": 1, "num_points": input_points},
                description=f"Embedding Conv1d(3→{embed_dim}, k=1)",
            ),
            OperatorRef(
                op_type="bn",
                params={"num_features": embed_dim, "num_points": input_points},
                description=f"Embedding BN({embed_dim})",
            ),
            OperatorRef(
                op_type="relu",
                params={"num_features": embed_dim, "num_points": input_points},
                description="Embedding ReLU",
            ),
        ],
        description=f"Input embedding: [B,3,{input_points}] → [B,{embed_dim},{input_points}]",
    ))

    # ------------------------------------------------------------------
    # 2-13. Stages 0-3: GAM + LFP + GFP (stage)
    # ------------------------------------------------------------------
    stage_points = _canonical_classification_stage_points(input_points, reducers)
    current_points = input_points
    current_channels = embed_dim

    for s in range(num_stages):
        out_points = stage_points[s]
        out_channels = current_channels * dim_expansion[s]
        k = k_neighbors[s]

        # --- GAM (identical to standard PointKAN) ---
        gam_ops = _build_gam_operators(
            input_points=current_points,
            output_points=out_points,
            ref_points=current_points,
            k=k,
            out_channels=out_channels,
            in_channels=current_channels,
        )

        layers.append(LayerDef(
            name=f"GAM-stage{s}",
            layer_type="GAM",
            operators=gam_ops,
            description=(
                f"GAM stage {s}: {current_points}→{out_points} points, "
                f"ch={current_channels}→{out_channels}, K={k}"
            ),
        ))

        # --- LFP (Elite-specific: Pre-Norm with Efficient_KAN) ---
        lfp_ops = _build_lfp_operators_elite(
            channels=out_channels,
            num_points=out_points,
            k_neighbors=k,
        )
        layers.append(LayerDef(
            name=f"LFP-stage{s}",
            layer_type="LFP",
            operators=lfp_ops,
            description=(
                f"LFP stage {s}: local + center paths, "
                f"Efficient_KAN({out_channels}→{out_channels}), "
                f"Pre-Norm residual: x + KAN(LayerNorm(x))"
            ),
        ))

        # --- GFP (stage-n, identical to standard PointKAN) ---
        gfp_ops = _build_gfp_operators(channels=out_channels, num_points=out_points)
        layers.append(LayerDef(
            name=f"stage-{s}",
            layer_type="stage",
            operators=gfp_ops,
            description=(
                f"GFP stage {s}: ConvBNReLURes1D({out_channels}→{out_channels}), "
                f"{out_points} points"
            ),
        ))

        current_points = out_points
        current_channels = out_channels

    # ------------------------------------------------------------------
    # 14. Global MaxPool：源码 F.adaptive_max_pool1d(x, 1) 在 classifier 前，形状 [B,512,64]→[B,512]
    # ------------------------------------------------------------------
    layers.append(LayerDef(
        name="global_maxpool",
        layer_type="global_maxpool",
        operators=[
            OperatorRef(
                op_type="adaptive_max_pool1d",
                params={
                    "output_size": 1,
                    "num_features": current_channels,
                    "num_points": current_points,
                },
                description=f"Global AdaptiveMaxPool1d(1): [B,{current_channels},{current_points}]→[B,{current_channels}]",
            )
        ],
        description=(
            f"Global MaxPool: [B,{current_channels},{current_points}] → "
            f"[B,{current_channels}]"
        ),
    ))

    # ------------------------------------------------------------------
    # 15. Classifier (identical to standard PointKAN)
    # ------------------------------------------------------------------
    fc_dim = current_channels  # 512
    mid_dim = fc_dim // 2       # 256
    classifier_ops = [
        OperatorRef(
            op_type="linear",
            params={"in_features": fc_dim, "out_features": fc_dim, "tokens": 1},
            description=f"Classifier FC1: Linear({fc_dim}→{fc_dim})",
        ),
        OperatorRef(
            op_type="batchnorm1d",
            params={"num_features": fc_dim},
            description=f"Classifier BN1d({fc_dim})",
        ),
        OperatorRef(
            op_type="relu",
            params={"num_features": fc_dim},
            description="Classifier ReLU [1]",
        ),
        # Dropout(0.5) — ignored (inference no-op)
        OperatorRef(
            op_type="linear",
            params={"in_features": fc_dim, "out_features": mid_dim, "tokens": 1},
            description=f"Classifier FC2: Linear({fc_dim}→{mid_dim})",
        ),
        OperatorRef(
            op_type="batchnorm1d",
            params={"num_features": mid_dim},
            description=f"Classifier BN1d({mid_dim})",
        ),
        OperatorRef(
            op_type="relu",
            params={"num_features": mid_dim},
            description="Classifier ReLU [2]",
        ),
        # Dropout(0.5) — ignored (inference no-op)
        OperatorRef(
            op_type="linear",
            params={"in_features": mid_dim, "out_features": num_classes, "tokens": 1},
            description=f"Classifier FC3: Linear({mid_dim}→{num_classes})",
        ),
    ]
    layers.append(LayerDef(
        name="classifier",
        layer_type="classifier",
        operators=classifier_ops,
        description=(
            f"Classification head: "
            f"FC({fc_dim}→{fc_dim})→BN→ReLU→"
            f"FC({fc_dim}→{mid_dim})→BN→ReLU→"
            f"FC({mid_dim}→{num_classes})"
        ),
    ))

    return layers


# ---------------------------------------------------------------------------
# ScanObjectNN classification (same architecture, 15 classes, use_xyz=False)
# ---------------------------------------------------------------------------

def create_pointkan_elite_scanobjectnn_classification(
    input_points: int = 1024,
    num_classes: int = 15,
    embed_dim: int = 32,
    dim_expansion: Optional[List[int]] = None,
    reducers: Optional[List[int]] = None,
    k_neighbors: Optional[List[int]] = None,
) -> List[LayerDef]:
    """
    Create PointKAN-Elite ScanObjectNN classification network layer sequence.

    Architecture is identical to the ModelNet40 classification variant
    (use_xyz=False, normalize="anchor"), only the class count differs.

    Default configuration (pointKANElite ScanObjectNN):
        input: [B, 3, 1024], 15 classes (ScanObjectNN)
        embed_dim=32, 4 stages, dim_expansion=[2,2,2,2],
        reducers=[2,2,2,2], k_neighbors=[24,24,24,24]

    Args:
        input_points: Number of input points.
        num_classes:  Number of output classes (default 15 for ScanObjectNN).
        embed_dim:    Initial embedding dimension.
        dim_expansion, reducers, k_neighbors: Per-stage hyperparameters.

    Returns:
        Ordered list of LayerDef defining the complete forward pass.
    """
    # The ScanObjectNN model is structurally identical to ModelNet40:
    # use_xyz=False → S-Pool width = 2*last_ch = out_ch (with dim_expansion=2)
    # → lfp_in_channels = out_channels (same as default).
    # Re-use create_pointkan_elite_classification with the appropriate num_classes.
    return create_pointkan_elite_classification(
        input_points=input_points,
        num_classes=num_classes,
        embed_dim=embed_dim,
        dim_expansion=dim_expansion,
        reducers=reducers,
        k_neighbors=k_neighbors,
    )


# ---------------------------------------------------------------------------
# Part segmentation (encoder-decoder with PointNetFeaturePropagation)
# ---------------------------------------------------------------------------

def _build_fp_operators(
    src_channels: int,
    skip_channels: int,
    fuse_out_channels: int,
    num_output_points: int,
    num_source_points: int,
    de_blocks: int = 1,
    stage_idx: int = 0,
    distance_reuse: bool = False,
) -> List[OperatorRef]:
    """
    Build operator list for a PointNetFeaturePropagation (FP/decoder) stage.

    Pipeline:
      - num_source_points > 1:
        1. KNN(k=3): find 3 nearest in xyz2 (coarser) for each point in xyz1 (finer)
           - distance_reuse=True: 跳过距离计算，从 DRAM 读取已存储的距离矩阵 + sort top-3
        2. InterpN3: gather + div + mul + add 统一插值引擎
      - num_source_points <= 1:
        1. Repeat/broadcast the lone source feature to all target points.
      [concat with skip connection — memory gather, no extra compute]
      3. fuse:     Conv1d(fuse_in → fuse_out) + BN + ReLU
                   fuse_in = src_channels + skip_channels
      4. extraction: de_blocks × ConvBNReLURes1D(fuse_out)

    Args:
        src_channels:       Channels of the coarser feature tensor (points2).
        skip_channels:      Channels of the skip-connection tensor (points1).
        fuse_out_channels:  Output channels after the fuse Conv1d.
        num_output_points:  Points in the finer set (xyz1, N).
        num_source_points:  Points in the coarser set (xyz2, S).
        de_blocks:          Number of ConvBNReLURes1D blocks in extraction.
        stage_idx:          Stage index for descriptive labels.
        distance_reuse:     Whether to reuse stored distance matrix from SA/GAM.
    """
    fuse_in_channels = src_channels + skip_channels
    ops: List[OperatorRef] = []

    if num_source_points > 1:
        # Standard decoder FP path: KNN + InterpN3.
        knn_params = {
            "input_points": num_source_points,
            "query_points": num_output_points,
            "k": 3,
            "distance_reuse": distance_reuse,
        }
        ops.append(OperatorRef(
            op_type="knn",
            params=knn_params,
            description=(
                f"FP-stage{stage_idx} interpolate KNN(k=3): "
                f"query={num_output_points}, ref={num_source_points}"
                + (" [distance_reuse]" if distance_reuse else "")
            ),
        ))

        # 2. InterpN3: unified gather + div/mul/add interpolation engine.
        ops.append(OperatorRef(
            op_type="interp_n3",
            params={
                "num_output_points": num_output_points,
                "src_channels": src_channels,
                "k": 3,
            },
            description=(
                f"FP-stage{stage_idx} InterpN3: gather+interp "
                f"{src_channels}ch@{num_output_points}pts, k=3"
            ),
        ))

    # 5. Fuse: Conv1d(fuse_in → fuse_out) + BN + ReLU
    ops.append(OperatorRef(
        op_type="conv1d",
        params={
            "in_channels": fuse_in_channels,
            "out_channels": fuse_out_channels,
            "kernel_size": 1,
            "num_points": num_output_points,
        },
        description=f"FP-stage{stage_idx} fuse Conv1d({fuse_in_channels}→{fuse_out_channels})",
    ))
    ops.append(OperatorRef(
        op_type="bn",
        params={"num_features": fuse_out_channels},
        description=f"FP-stage{stage_idx} fuse BN({fuse_out_channels})",
    ))
    ops.append(OperatorRef(
        op_type="relu",
        params={"num_features": fuse_out_channels},
        description=f"FP-stage{stage_idx} fuse ReLU",
    ))

    # 6. Extraction: de_blocks × ConvBNReLURes1D(fuse_out_channels)
    for b in range(de_blocks):
        for op in _build_gfp_operators(channels=fuse_out_channels, num_points=num_output_points):
            ops.append(OperatorRef(
                op_type=op.op_type,
                params=dict(op.params),
                description=f"FP-stage{stage_idx} extraction blk{b}: {op.description}",
            ))

    return ops


def create_pointkan_elite_partseg(
    input_points: int = 2048,
    num_classes: int = 50,
    embed_dim: int = 32,
    dim_expansion: Optional[List[int]] = None,
    reducers: Optional[List[int]] = None,
    k_neighbors: Optional[List[int]] = None,
    de_dims: Optional[List[int]] = None,
    de_blocks: Optional[List[int]] = None,
    gmp_dim: int = 64,
    cls_dim: int = 64,
    enable_distance_reuse: bool = False,
) -> List[LayerDef]:
    """
    Create PointKAN-Elite part-segmentation network layer sequence.

    Corresponds to ``PointKAN-elite/part_segmentation/model/pointKANElite.py``.

    Default configuration (pointKANElite part-seg factory):
        input: [B, 6, 2048]  (xyz + normal vectors)
        50 ShapeNetPart classes, embed_dim=32,
        4 encoder stages: reducers=[4,4,4,4], k_neighbors=[32,32,32,32],
        use_xyz=True (xyz appended to grouped features in GAM/LFP),
        4 decoder stages (PointNetFeaturePropagation), de_blocks=[4,4,4,4].

    Returns:
        Ordered list of LayerDef.
    """
    if dim_expansion is None:
        dim_expansion = [2, 2, 2, 2]
    if reducers is None:
        reducers = [4, 4, 4, 4]
    if k_neighbors is None:
        k_neighbors = [32, 32, 32, 32]
    if de_dims is None:
        de_dims = [512, 256, 128, 128]
    if de_blocks is None:
        de_blocks = [4, 4, 4, 4]

    num_stages = len(dim_expansion)
    layers: List[LayerDef] = []

    # ------------------------------------------------------------------
    # 1. Embedding: Conv1d(6→embed_dim) + BN + ReLU  (input is xyz+normal)
    # ------------------------------------------------------------------
    layers.append(LayerDef(
        name="embedding",
        layer_type="embedding",
        operators=[
            OperatorRef(
                op_type="conv1d",
                params={"in_channels": 6, "out_channels": embed_dim,
                        "kernel_size": 1, "num_points": input_points},
                description=f"Embedding Conv1d(6→{embed_dim}, k=1) [xyz+normal]",
            ),
            OperatorRef(
                op_type="bn",
                params={"num_features": embed_dim, "num_points": input_points},
                description=f"Embedding BN({embed_dim})",
            ),
            OperatorRef(
                op_type="relu",
                params={"num_features": embed_dim, "num_points": input_points},
                description="Embedding ReLU",
            ),
        ],
        description=f"Input embedding: [B,6,{input_points}] → [B,{embed_dim},{input_points}]",
    ))

    # ------------------------------------------------------------------
    # 2-13. Encoder: 4 stages of GAM + LFP + GFP (use_xyz=True)
    # ------------------------------------------------------------------
    stage_points = _canonical_partseg_stage_points(input_points, reducers)
    current_points = input_points
    current_channels = embed_dim
    en_dims = [current_channels]   # [32]  (grows as stages run)
    enc_pts = [current_points]     # track point counts for decoder

    for s in range(num_stages):
        out_points = stage_points[s]
        out_channels = current_channels * dim_expansion[s]
        k = k_neighbors[s]

        # use_xyz=True: spool_in_channels = 2*current_channels + 3
        # base_v3: GAM-0 writes distance matrix to DRAM; GAM-1+ read sub-matrix
        gam_ops = _build_gam_operators(
            input_points=current_points,
            output_points=out_points,
            ref_points=current_points,
            k=k,
            out_channels=out_channels,
            in_channels=current_channels,
            use_xyz=True,
            dist_matrix_dram_write=(enable_distance_reuse and s == 0),
            distance_reuse=(enable_distance_reuse and s > 0),
        )
        _flags = []
        if enable_distance_reuse and s == 0:
            _flags.append("dist_matrix_dram_write")
        if enable_distance_reuse and s > 0:
            _flags.append("distance_reuse")
        layers.append(LayerDef(
            name=f"GAM-stage{s}",
            layer_type="GAM",
            operators=gam_ops,
            description=(
                f"GAM stage{s} (use_xyz=True): {current_points}→{out_points} pts, "
                f"ch={current_channels}→{out_channels}, K={k}"
                + (f" [{','.join(_flags)}]" if _flags else "")
            ),
        ))

        # LFP: transfer in_channels = 2*current_channels + 3  (use_xyz=True)
        lfp_in_channels = 2 * current_channels + 3
        lfp_ops = _build_lfp_operators_elite(
            channels=out_channels,
            num_points=out_points,
            k_neighbors=k,
            lfp_in_channels=lfp_in_channels,
        )
        layers.append(LayerDef(
            name=f"LFP-stage{s}",
            layer_type="LFP",
            operators=lfp_ops,
            description=(
                f"LFP stage{s} (use_xyz=True): "
                f"transfer Conv1d({lfp_in_channels}→{out_channels}), "
                f"Efficient_KAN({out_channels}→{out_channels}), Pre-Norm"
            ),
        ))

        gfp_ops = _build_gfp_operators(channels=out_channels, num_points=out_points)
        layers.append(LayerDef(
            name=f"GFP-stage{s}",
            layer_type="stage",
            operators=gfp_ops,
            description=f"GFP stage{s}: ConvBNReLURes1D({out_channels}), {out_points} pts",
        ))

        current_points = out_points
        current_channels = out_channels
        en_dims.append(current_channels)
        enc_pts.append(current_points)

    # en_dims = [32, 64, 128, 256, 512]  enc_pts = [2048, 512, 128, 32, 8]

    # ------------------------------------------------------------------
    # 14-17. Decoder: PointNetFeaturePropagation × 4
    #
    # After reversals in the model:
    #   x_list (reversed)   = [512ch@8pts, 256ch@32pts, 128ch@128pts, 64ch@512pts, 32ch@2048pts]
    #   xyz_list (reversed) = [8pts, 32pts, 128pts, 512pts, 2048pts]
    #   en_dims (reversed)  = [512, 256, 128, 64, 32]
    #   de_dims after insert = [512, 512, 256, 128, 128]
    #
    # Decode i: xyz1=enc_pts_rev[i+1], xyz2=enc_pts_rev[i]
    #           src_ch = de_dims_full[i], skip_ch = en_dims_rev[i+1]
    #           fuse_in = src_ch + skip_ch, fuse_out = de_dims_full[i+1]
    # ------------------------------------------------------------------
    en_dims_rev = list(reversed(en_dims))           # [512, 256, 128, 64, 32]
    enc_pts_rev = list(reversed(enc_pts))           # [8, 32, 128, 512, 2048]
    de_dims_full = [en_dims_rev[0]] + list(de_dims)  # [512, 512, 256, 128, 128]

    for i in range(len(de_dims)):
        xyz1_pts = enc_pts_rev[i + 1]    # finer (query)
        xyz2_pts = enc_pts_rev[i]        # coarser (reference)
        src_ch = de_dims_full[i]         # current decoder feature channels
        skip_ch = en_dims_rev[i + 1]     # skip connection channels
        fuse_out = de_dims_full[i + 1]   # decoder output channels
        n_deblks = de_blocks[i]

        fp_ops = _build_fp_operators(
            src_channels=src_ch,
            skip_channels=skip_ch,
            fuse_out_channels=fuse_out,
            num_output_points=xyz1_pts,
            num_source_points=xyz2_pts,
            de_blocks=n_deblks,
            stage_idx=i,
            distance_reuse=enable_distance_reuse,
        )
        layers.append(LayerDef(
            name=f"FP-stage{i}",
            layer_type="FP",
            operators=fp_ops,
            description=(
                f"FP stage{i}: interp {xyz2_pts}→{xyz1_pts} pts, "
                f"fuse Conv1d({src_ch + skip_ch}→{fuse_out}), "
                f"extraction {n_deblks}×ConvBNReLURes1D({fuse_out})"
            ),
        ))

    # After decoder: x has shape [B, de_dims_full[-1]=128, input_points=2048]
    final_dec_ch = de_dims_full[-1]   # 128

    # ------------------------------------------------------------------
    # 18. Global context: gmp_map_list (one per encoder stage incl. embedding)
    #     + gmp_map_end
    #     en_dims_rev = [512, 256, 128, 64, 32], enc_pts_rev = [8, 32, 128, 512, 2048]
    # ------------------------------------------------------------------
    gmp_ops: List[OperatorRef] = []
    for i, (en_ch, n_pts) in enumerate(zip(en_dims_rev, enc_pts_rev)):
        gmp_ops.append(OperatorRef(
            op_type="conv1d",
            params={"in_channels": en_ch, "out_channels": gmp_dim,
                    "kernel_size": 1, "num_points": n_pts},
            description=f"GMP gmp_map[{i}] Conv1d({en_ch}→{gmp_dim}) @{n_pts}pts",
        ))
        gmp_ops.append(OperatorRef(
            op_type="bn",
            params={"num_features": gmp_dim},
            description=f"GMP gmp_map[{i}] BN({gmp_dim})",
        ))
        gmp_ops.append(OperatorRef(
            op_type="relu",
            params={"num_features": gmp_dim},
            description=f"GMP gmp_map[{i}] ReLU",
        ))
        gmp_ops.append(OperatorRef(
            op_type="adaptive_max_pool1d",
            params={"output_size": 1, "num_features": gmp_dim, "num_points": n_pts},
            description=f"GMP gmp_map[{i}] AdaptiveMaxPool1d(1) @{n_pts}pts",
        ))
    # gmp_map_end: concat of 5 × [B, gmp_dim, 1] → Conv1d(gmp_dim*5→gmp_dim) + BN + ReLU
    gmp_concat_ch = gmp_dim * len(en_dims_rev)
    gmp_ops.append(OperatorRef(
        op_type="conv1d",
        params={"in_channels": gmp_concat_ch, "out_channels": gmp_dim,
                "kernel_size": 1, "num_points": 1},
        description=f"GMP gmp_map_end Conv1d({gmp_concat_ch}→{gmp_dim}) @1pt",
    ))
    gmp_ops.append(OperatorRef(
        op_type="bn",
        params={"num_features": gmp_dim},
        description=f"GMP gmp_map_end BN({gmp_dim})",
    ))
    gmp_ops.append(OperatorRef(
        op_type="relu",
        params={"num_features": gmp_dim},
        description="GMP gmp_map_end ReLU",
    ))
    layers.append(LayerDef(
        name="global_context",
        layer_type="global_context",
        operators=gmp_ops,
        description=(
            f"Global context: {len(en_dims_rev)}×gmp_map Conv1d→BN→ReLU→AMP(1), "
            f"gmp_map_end Conv1d({gmp_concat_ch}→{gmp_dim})"
        ),
    ))

    # ------------------------------------------------------------------
    # 19. Class token: cls_map (two Conv1d layers on [B,16,1])
    # ------------------------------------------------------------------
    cls_ops: List[OperatorRef] = [
        OperatorRef(
            op_type="conv1d",
            params={"in_channels": 16, "out_channels": cls_dim,
                    "kernel_size": 1, "num_points": 1},
            description=f"CLS cls_map[0] Conv1d(16→{cls_dim})",
        ),
        OperatorRef(
            op_type="bn",
            params={"num_features": cls_dim},
            description=f"CLS cls_map[0] BN({cls_dim})",
        ),
        OperatorRef(
            op_type="relu",
            params={"num_features": cls_dim},
            description="CLS cls_map[0] ReLU",
        ),
        OperatorRef(
            op_type="conv1d",
            params={"in_channels": cls_dim, "out_channels": cls_dim,
                    "kernel_size": 1, "num_points": 1},
            description=f"CLS cls_map[1] Conv1d({cls_dim}→{cls_dim})",
        ),
        OperatorRef(
            op_type="bn",
            params={"num_features": cls_dim},
            description=f"CLS cls_map[1] BN({cls_dim})",
        ),
        OperatorRef(
            op_type="relu",
            params={"num_features": cls_dim},
            description="CLS cls_map[1] ReLU",
        ),
    ]
    layers.append(LayerDef(
        name="cls_token",
        layer_type="cls_token",
        operators=cls_ops,
        description=f"Class token: cls_map 2×[Conv1d+BN+ReLU], 16→{cls_dim}→{cls_dim}",
    ))

    # ------------------------------------------------------------------
    # 20. Classifier: Conv1d(gmp_dim+cls_dim+final_dec_ch → 128) + BN +
    #                  Conv1d(128 → num_classes)  @input_points points
    #     Input: cat(x[final_dec_ch], global[gmp_dim], cls[cls_dim]) broadcast to input_points
    # ------------------------------------------------------------------
    clf_in = gmp_dim + cls_dim + final_dec_ch   # 64+64+128=256
    clf_ops: List[OperatorRef] = [
        OperatorRef(
            op_type="conv1d",
            params={"in_channels": clf_in, "out_channels": 128,
                    "kernel_size": 1, "num_points": input_points},
            description=f"Classifier Conv1d({clf_in}→128, k=1)",
        ),
        OperatorRef(
            op_type="bn",
            params={"num_features": 128},
            description="Classifier BN(128)",
        ),
        # Dropout — inference no-op, omitted
        OperatorRef(
            op_type="conv1d",
            params={"in_channels": 128, "out_channels": num_classes,
                    "kernel_size": 1, "num_points": input_points},
            description=f"Classifier Conv1d(128→{num_classes}, k=1)",
        ),
    ]
    layers.append(LayerDef(
        name="classifier",
        layer_type="classifier",
        operators=clf_ops,
        description=(
            f"Per-point segmentation head: "
            f"Conv1d({clf_in}→128)→BN→Conv1d(128→{num_classes}), "
            f"{input_points} pts"
        ),
    ))

    return layers


# ---------------------------------------------------------------------------
# Convenience accessor
# ---------------------------------------------------------------------------

def get_pointkan_elite_info() -> Dict[str, Any]:
    """
    Get information about PointKAN-Elite network configuration.

    Returns:
        Dictionary with configuration information.
    """
    return {
        "name": "PointKAN-Elite",
        "task": "classification",
        "dataset": "ModelNet40",
        "input_shape": "[B, 3, 1024]",
        "num_classes": 40,
        "embed_dim": 32,
        "stages": 4,
        "dim_expansion": [2, 2, 2, 2],
        "reducers": [2, 2, 2, 2],
        "k_neighbors": [24, 24, 24, 24],
        "kan_variant": "elite (Efficient_KAN 展开为 LFP 内显式 linear: operation1/2.fc1, fc2)",
        "lfp_norm_order": "Pre-Norm (LayerNorm → Efficient_KAN)",
        "differences_from_pointkan": [
            "LFP uses Pre-Norm: x = x + operation(LayerNorm(x))",
            "LFP 内 Efficient_KAN 展开为 4 个 linear（operation1.fc1/fc2, operation2.fc1/fc2），与真实前向 36 层严格对齐",
            "No DWConv inside KAN module",
            "Rational activation P(x)/Q(x) instead of B-spline basis functions",
        ],
        "operators_used": [
            "conv1d", "bn", "relu", "fps", "knn", "affine_norm", "s-pool",
            "layernorm", "linear", "batchnorm1d",
            "adaptive_max_pool1d", "add", "mul",
        ],
    }


# ---------------------------------------------------------------------------
# Default instance for quick access
# ---------------------------------------------------------------------------

POINTKAN_ELITE_LAYERS = create_pointkan_elite_classification()
"""Default PointKAN-Elite classification layer sequence (1024 points, 40 classes)."""
