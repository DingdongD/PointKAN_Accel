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


# ---------------------------------------------------------------------------
# LFP operator builder (Elite-specific: Pre-Norm)
# ---------------------------------------------------------------------------

def _build_lfp_operators_elite(channels: int, num_points: int) -> List[OperatorRef]:
    """
    Build operator list for a PointKAN-Elite LFP layer.

    Pre-Norm residual: x = x + Efficient_KAN(LayerNorm(x))
    Efficient_KAN 展开为显式 linear：fc1(in→hidden)、fc2(hidden→out)，与真实模型
    LFP_blocks_list.*.operation1.0.fc1/fc2、operation2.0.fc1/fc2 严格对齐，便于 MNSIM/仿真 36 层一致。

    PointKAN-Elite LFP structure (per stage):
      Local path:  transfer1 → LayerNorm → [fc1, fc2] (operation1) → Add(residual) → AdaptiveMaxPool1d(1)
      Center path: transfer2 → LayerNorm → [fc1, fc2] (operation2) → Add(residual)
      Fusion:      local + center (element-wise add)
    """
    ops = []

    # --- Local path ---
    # transfer1: Conv1d + BN + ReLU
    ops.append(OperatorRef(
        op_type="conv1d",
        params={"in_channels": channels, "out_channels": channels, "kernel_size": 1, "num_points": num_points},
        description=f"LFP transfer1: Conv1d({channels}→{channels}, k=1)",
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
        params={"in_features": channels, "out_features": channels, "tokens": num_points},
        description=f"LFP local operation1.fc1 Linear({channels}→{channels})",
    ))
    ops.append(OperatorRef(
        op_type="kan",
        params={"in_features": channels, "out_features": channels, "num_points": num_points, "variant": "elite"},
        description=f"LFP local operation1 KAT/GRF ({channels}→{channels}, rational activation) [act2 only]",
    ))
    ops.append(OperatorRef(
        op_type="linear",
        params={"in_features": channels, "out_features": channels, "tokens": num_points},
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
        params={"in_channels": channels, "out_channels": channels, "kernel_size": 1, "num_points": num_points},
        description=f"LFP transfer2: Conv1d({channels}→{channels}, k=1)",
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
        params={"in_features": channels, "out_features": channels, "tokens": num_points},
        description=f"LFP center operation2.fc1 Linear({channels}→{channels})",
    ))
    ops.append(OperatorRef(
        op_type="kan",
        params={"in_features": channels, "out_features": channels, "num_points": num_points, "variant": "elite"},
        description=f"LFP center operation2 KAT/GRF ({channels}→{channels}, rational activation) [act2 only]",
    ))
    ops.append(OperatorRef(
        op_type="linear",
        params={"in_features": channels, "out_features": channels, "tokens": num_points},
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
    current_points = input_points
    current_channels = embed_dim

    for s in range(num_stages):
        out_points = current_points // reducers[s]
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
        lfp_ops = _build_lfp_operators_elite(channels=out_channels, num_points=out_points)
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
