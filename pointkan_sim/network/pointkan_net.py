"""
PointKAN Network Definition
============================

Manual definition of PointKAN network layers for CIM hardware mapping.
Defines the complete forward-pass layer sequence with per-layer operator lists.

Architecture overview (PointKAN classification, ModelNet40):
    Input [B, 3, 1024]
     → Embedding: Conv1d(3→32) + BN + ReLU
     → Stage 0~3 (x4, each: points halved, channels doubled):
         ├ GAM: FPS → KNN → AffineNorm → Concat → S-Pool
         ├ LFP: Conv1d+BN+ReLU → KANLayer → LayerNorm → AdaptiveMaxPool1d (local);
                 transfer2 → KAN → LayerNorm (center); Add
         └ GFP: Conv1d→BN→ReLU → Conv1d→BN → Add(residual) → ReLU
     → Global MaxPool
     → Classifier: Linear→BN1d→ReLU (x2) → Linear

"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional


# ---------------------------------------------------------------------------
# Layer & Operator reference data structures
# ---------------------------------------------------------------------------

@dataclass
class OperatorRef:
    """
    Reference to an operator instance within a network layer.

    Attributes:
        op_type: Operator type name. One of:
            "conv1d", "bn", "relu", "fps", "knn", "affine_norm", "s-pool",
            "layernorm", "linear", "batchnorm1d", "kan",
            "adaptive_max_pool1d", "add", "mul"
        params: Operator-specific parameter dict
        description: Human-readable description of this operator instance
    """
    op_type: str
    params: Dict[str, Any] = field(default_factory=dict)
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "op_type": self.op_type,
            "params": self.params,
            "description": self.description,
        }


@dataclass
class LayerDef:
    """
    Definition of a network layer containing an ordered list of operators.

    Attributes:
        name: Layer name following naming convention
            (e.g., "embedding", "GAM-stage0", "LFP-stage0", "stage-0",
             "global_maxpool", "classifier")
        layer_type: Layer type category
            ("embedding", "GAM", "LFP", "stage", "global_maxpool", "classifier")
        operators: Ordered list of operators executed in this layer
        description: Human-readable description
    """
    name: str
    layer_type: str
    operators: List[OperatorRef] = field(default_factory=list)
    description: str = ""

    def get_operator_types(self) -> List[str]:
        """Get ordered list of operator type names."""
        return [op.op_type for op in self.operators]

    def get_operator_count(self) -> int:
        return len(self.operators)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "layer_type": self.layer_type,
            "operators": [op.to_dict() for op in self.operators],
            "description": self.description,
        }


# ---------------------------------------------------------------------------
# PointKAN configuration
# ---------------------------------------------------------------------------

@dataclass
class PointKANConfig:
    """
    PointKAN network configuration.

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
    name: str = "PointKAN"
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
# Helper: build per-stage operator lists
# ---------------------------------------------------------------------------

def _build_gam_operators(
    input_points: int,
    output_points: int,
    ref_points: int,
    k: int,
    out_channels: int,
    in_channels: int,
) -> List[OperatorRef]:
    """
    Build operator list for a GAM (Geometric Affine Module) layer.

    GAM pipeline: FPS → KNN → (Gather → AffineNorm → Concat) → S-Pool
    AffineNorm: (x-mean)/(std+eps) then alpha*x+beta (fixed at inference).
    """
    return [
        OperatorRef(
            op_type="fps",
            params={"input_points": ref_points, "output_points": output_points},
            description=f"FPS: {ref_points}→{output_points} points",
        ),
        OperatorRef(
            op_type="knn",
            params={
                "input_points": ref_points,
                "query_points": output_points,
                "k": k,
            },
            description=f"KNN: query={output_points}, ref={ref_points}, K={k}",
        ),
        OperatorRef(
            op_type="affine_norm",
            params={
                "num_features": in_channels,
                "num_points": output_points,
                "k_neighbors": k,
                "normalize": "anchor",
                "use_xyz": False,
            },
            description=f"GAM AffineNorm (inference fixed): norm then alpha*x+beta, dim={in_channels}",
        ),
        OperatorRef(
            op_type="s-pool",
            params={
                "in_channels": out_channels,
                "num_points": output_points,
                "k_neighbors": k,
            },
            description=f"S-Pool: [{output_points},{k},{out_channels}]→[{out_channels},{output_points}]",
        ),
    ]


def _build_lfp_operators_pointkan(
    channels: int,
    num_points: int,
) -> List[OperatorRef]:
    """
    Build operator list for a PointKAN LFP (Local Feature Processing) layer.

    PointKAN LFP structure (per stage):
      Local path:  transfer1(Conv1d+BN+ReLU) → KANLayer → LayerNorm → AdaptiveMaxPool1d(1)
      Center path: transfer2(Conv1d+BN+ReLU) → KANLayer → LayerNorm
      Fusion:      local + center (element-wise add)
    Operator order follows forward execution strictly.
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
    # KANLayer (black-box, standard B-spline variant)
    ops.append(OperatorRef(
        op_type="kan",
        params={"in_features": channels, "out_features": channels, "num_points": num_points, "variant": "standard"},
        description=f"LFP local KANLayer({channels}→{channels})",
    ))
    # LayerNorm
    ops.append(OperatorRef(
        op_type="layernorm",
        params={"normalized_shape": channels, "num_points": num_points},
        description=f"LFP local LayerNorm({channels})",
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
    # KANLayer (black-box, standard B-spline variant)
    ops.append(OperatorRef(
        op_type="kan",
        params={"in_features": channels, "out_features": channels, "num_points": num_points, "variant": "standard"},
        description=f"LFP center KANLayer({channels}→{channels})",
    ))
    # LayerNorm
    ops.append(OperatorRef(
        op_type="layernorm",
        params={"normalized_shape": channels, "num_points": num_points},
        description=f"LFP center LayerNorm({channels})",
    ))
    # Fusion: element-wise add (local + center)
    ops.append(OperatorRef(
        op_type="add",
        params={"num_features": channels, "num_points": num_points},
        description="LFP fusion: local + center",
    ))

    return ops


def _build_gfp_operators(channels: int, num_points: int) -> List[OperatorRef]:
    """
    Build operator list for a GFP (Global Feature Processing) layer.

    GFP = ConvBNReLURes1D: two Conv1d-BN blocks with residual.
    Structure: Conv1d → BN → ReLU → Conv1d → BN → Add(residual) → ReLU
    """
    return [
        OperatorRef(
            op_type="conv1d",
            params={"in_channels": channels, "out_channels": channels, "kernel_size": 1, "num_points": num_points},
            description=f"GFP Conv1d({channels}→{channels}, k=1) [layer 1]",
        ),
        OperatorRef(
            op_type="bn",
            params={"num_features": channels},
            description=f"GFP BN({channels}) [layer 1]",
        ),
        OperatorRef(
            op_type="relu",
            params={"num_features": channels},
            description="GFP ReLU [layer 1]",
        ),
        OperatorRef(
            op_type="conv1d",
            params={"in_channels": channels, "out_channels": channels, "kernel_size": 1, "num_points": num_points},
            description=f"GFP Conv1d({channels}→{channels}, k=1) [layer 2]",
        ),
        OperatorRef(
            op_type="bn",
            params={"num_features": channels},
            description=f"GFP BN({channels}) [layer 2]",
        ),
        OperatorRef(
            op_type="add",
            params={"residual": True, "num_features": channels, "num_points": num_points},
            description="GFP residual add",
        ),
        OperatorRef(
            op_type="relu",
            params={"num_features": channels},
            description="GFP final ReLU (act after residual)",
        ),
    ]


# ---------------------------------------------------------------------------
# Network creation functions
# ---------------------------------------------------------------------------

def create_pointkan_classification(
    input_points: int = 1024,
    num_classes: int = 40,
    embed_dim: int = 32,
    dim_expansion: Optional[List[int]] = None,
    reducers: Optional[List[int]] = None,
    k_neighbors: Optional[List[int]] = None,
) -> List[LayerDef]:
    """
    Create PointKAN classification network layer sequence.

    Default configuration (pointKAN factory function):
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

        # --- GAM ---
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

        # --- LFP ---
        lfp_ops = _build_lfp_operators_pointkan(channels=out_channels, num_points=out_points)
        layers.append(LayerDef(
            name=f"LFP-stage{s}",
            layer_type="LFP",
            operators=lfp_ops,
            description=(
                f"LFP stage {s}: local + center paths, "
                f"KANLayer({out_channels}→{out_channels}), "
                f"Post-Norm residual"
            ),
        ))

        # --- GFP (stage-n) ---
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
    # 14. Global MaxPool：等价为 adaptive_max_pool（与 LFP 等 pool 同类），计入 latency/area/power
    # ------------------------------------------------------------------
    # 源码 F.adaptive_max_pool1d(x, 1)，形状 [B, C, L] → [B, C]
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
    # 15. Classifier: Linear→BN1d→ReLU → Linear→BN1d→ReLU → Linear
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

def get_pointkan_info() -> Dict[str, Any]:
    """
    Get information about PointKAN network configuration.

    Returns:
        Dictionary with configuration information.
    """
    return {
        "name": "PointKAN",
        "task": "classification",
        "dataset": "ModelNet40",
        "input_shape": "[B, 3, 1024]",
        "num_classes": 40,
        "embed_dim": 32,
        "stages": 4,
        "dim_expansion": [2, 2, 2, 2],
        "reducers": [2, 2, 2, 2],
        "k_neighbors": [24, 24, 24, 24],
        "kan_variant": "standard (B-spline KAN + DWConv + residual)",
        "lfp_norm_order": "Post-Norm (KAN → LayerNorm)",
        "operators_used": [
            "conv1d", "bn", "relu", "fps", "knn", "affine_norm", "s-pool",
            "layernorm", "kan", "linear", "batchnorm1d",
            "adaptive_max_pool1d", "add", "mul",
        ],
    }


# ---------------------------------------------------------------------------
# Default instance for quick access
# ---------------------------------------------------------------------------

POINTKAN_LAYERS = create_pointkan_classification()
"""Default PointKAN classification layer sequence (1024 points, 40 classes)."""
