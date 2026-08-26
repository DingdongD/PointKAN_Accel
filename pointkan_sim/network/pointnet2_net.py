"""
PointNet++ Network Definitions
================================

Manual definitions of PointNet++ network layer sequences for CIM hardware mapping.
Supports three variants from ISEDA26/model_back.py:

  - PointNet2ClsSSG   : ModelNet40 classification (SSG, no MSG)
  - PointNet2PartSegSSG: ShapeNetPart part segmentation (SSG)
  - PointNet2SemSSG   : S3DIS semantic segmentation (SSG, 4 SA + 4 FP)

Architecture references (model_back.py):


PointNet2ClsSSG:
    Input [B, 3, 1024]
    → SA1: FPS(1024→512) + BallQuery(k=32) + Conv1d[3→64→64→128] + MaxPool
    → SA2: FPS(512→128) + BallQuery(k=64) + Conv1d[131→128→128→256] + MaxPool
    → SA3: GroupAll + Conv1d[259→256→512→1024] + MaxPool
    → FC: Linear(1024→512)→BN→ReLU → Linear(512→256)→BN→ReLU → Linear(256→num_classes)

PointNet2PartSegSSG:
    Input [B, 3, 2048]  (no-normal mode)
    → SA1: FPS(2048→512) + BallQuery(k=32) + Conv1d[6→64→64→128] + MaxPool
    → SA2: FPS(512→128) + BallQuery(k=64) + Conv1d[131→128→128→256] + MaxPool
    → SA3: GroupAll + Conv1d[259→256→512→1024] + MaxPool
    → FP1: KNN-interp(1→128) + Conv1d[1280→256→256]
    → FP2: KNN-interp(128→512) + Conv1d[384→256→128]
    → FP3: KNN-interp(512→2048) + Conv1d[150→128→128→128]
    → Head: Conv1d(128→128)→BN→ReLU → Conv1d(128→num_classes)

PointNet2SemSSG:
    Input [B, 12, 4096]  (xyz + rgb + normalised_xyz + extra)
    → SA1: FPS(4096→1024) + BallQuery(k=32) + Conv1d[12→32→32→64] + MaxPool
    → SA2: FPS(1024→256) + BallQuery(k=32) + Conv1d[67→64→64→128] + MaxPool
    → SA3: FPS(256→64) + BallQuery(k=32) + Conv1d[131→128→128→256] + MaxPool
    → SA4: FPS(64→16) + BallQuery(k=32) + Conv1d[259→256→256→512] + MaxPool
    → FP1: KNN-interp(16→64) + Conv1d[768→256→256]
    → FP2: KNN-interp(64→256) + Conv1d[384→256→256]
    → FP3: KNN-interp(256→1024) + Conv1d[320→256→128]
    → FP4: KNN-interp(1024→4096) + Conv1d[128→128→128→128]
    → Head: Conv1d(128→128)→BN→ReLU → Conv1d(128→num_classes)

Operator mapping:
  SA Conv2d(k=1) layers → conv2d with num_points = npoint * nsample
    (Conv2d kernel=1 on [B, C, nsample, npoint] maps to C_in * C_out MACs
     at each of npoint*nsample spatial locations)
  SA MaxPool (over nsample) → sa_maxpool(output_size=1, num_points=npoint*nsample)
  SA3 GroupAll Conv2d → conv1d with num_points = N_in (all points as one group)
  FP interpolation (3-NN) → knn(k=3, query_points=N_target, input_points=N_src) + interp_n3
  FP Conv1d MLP → conv1d with num_points = N_target
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

from .pointkan_net import OperatorRef, LayerDef


# ---------------------------------------------------------------------------
# Helper: build SA (Set Abstraction) operators
# ---------------------------------------------------------------------------

def _build_sa_operators(
    sa_index: int,
    in_points: int,
    out_points: int,
    nsample: int,
    in_channel: int,
    mlp: List[int],
    group_all: bool = False,
    dist_matrix_dram_write: bool = False,
    distance_reuse: bool = False,
) -> List[OperatorRef]:
    """
    Build operators for one PointNetSetAbstraction layer.

    Args:
        sa_index:   Stage index (0-based), for description labels.
        in_points:  Input point count before FPS.
        out_points: Output point count after FPS (ignored when group_all=True).
        nsample:    Neighbourhood sample count per query point.
        in_channel: MLP input channels (xyz concat with features).
        mlp:        Per-layer output channel list for the shared MLP.
        group_all:  When True, groups all in_points into one super-group
                    (SA3 GlobalAbstraction); no FPS/KNN emitted.
        dist_matrix_dram_write: If True, KNN writes distance matrix to DRAM for FP reuse.
        distance_reuse: If True, KNN skips distance computation and reads
                 sub-matrix from DRAM-stored global distance matrix.
    Returns:
        Ordered list of OperatorRef for one SA stage.
    """
    ops: List[OperatorRef] = []

    if group_all:
        npoints_mlp = in_points
        nq = 1
    else:
        ops.append(OperatorRef(
            op_type="fps",
            params={"input_points": in_points, "output_points": out_points},
            description=f"SA{sa_index} FPS: {in_points}→{out_points} points",
        ))
        knn_params: Dict[str, Any] = {
            "input_points": in_points,
            "query_points": out_points,
            "k": nsample,
        }
        if dist_matrix_dram_write:
            knn_params["dist_matrix_dram_write"] = True
        if distance_reuse:
            knn_params["distance_reuse"] = True
        _flags = []
        if dist_matrix_dram_write:
            _flags.append("dist_matrix_dram_write")
        if distance_reuse:
            _flags.append("distance_reuse")
        ops.append(OperatorRef(
            op_type="ballquery",
            params=knn_params,
            description=(
                f"SA{sa_index} BallQuery: query={out_points}, ref={in_points}, k={nsample}"
                + (f" [{','.join(_flags)}]" if _flags else "")
            ),
        ))
        npoints_mlp = out_points * nsample  # total (query, neighbour) pairs
        nq = out_points

    # Shared MLP: Conv2d(in_ch, mlp[j], 1) on [B, C, nsample, nq]
    # → represented as Conv1d with num_points = nq * nsample
    prev_ch = in_channel
    for j, out_ch in enumerate(mlp):
        conv_params: Dict[str, Any] = {
            "in_channels": prev_ch,
            "out_channels": out_ch,
            "kernel_size": 1,
            "num_points": npoints_mlp,
        }
        if j == 0:
            _xyz_ch = 3
            _feat_ch = max(0, prev_ch - _xyz_ch)
            conv_params["channel_semantics"] = {
                "relative_xyz": _xyz_ch,
                "grouped_features": _feat_ch,
                "has_relative_xyz": True,
                "has_concat": _feat_ch > 0,
                "source": "sample_and_group",
            }
        ops.append(OperatorRef(
            op_type="conv2d",
            params=conv_params,
            description=f"SA{sa_index} MLP[{j}]: Conv2d(k=1, {prev_ch}→{out_ch}), {npoints_mlp} elems",
        ))
        ops.append(OperatorRef(
            op_type="bn",
            params={"num_features": out_ch, "num_points": npoints_mlp},
            description=f"SA{sa_index} MLP[{j}]: BN({out_ch})",
        ))
        ops.append(OperatorRef(
            op_type="relu",
            params={"num_features": out_ch, "num_points": npoints_mlp},
            description=f"SA{sa_index} MLP[{j}]: ReLU",
        ))
        prev_ch = out_ch

    # MaxPool over nsample dimension: [B, C, nsample, nq] → [B, C, nq]
    # group_all 时 nq=1，有效 nsample 为全部 in_points
    _nsample_eff = in_points if group_all else nsample
    ops.append(OperatorRef(
        op_type="sa_maxpool",
        params={
            "output_size": 1,
            "num_features": mlp[-1],
            "num_points": npoints_mlp,   # 保留用于向后兼容
            "nq": nq,                    # query 点数（输出点数）
            "nsample": _nsample_eff,     # 每 query 的邻居数
        },
        description=f"SA{sa_index} MaxPool(nsample={_nsample_eff}→1): {mlp[-1]}ch, {nq} pts",
    ))

    return ops


# ---------------------------------------------------------------------------
# Helper: build FP (Feature Propagation) operators
# ---------------------------------------------------------------------------

def _build_fp_operators(
    fp_index: int,
    n_target: int,
    n_src: int,
    in_channel: int,
    mlp: List[int],
    distance_reuse: bool = False,
    src_channels: int = 0,
) -> List[OperatorRef]:
    """
    Build operators for one PointNetFeaturePropagation layer.

    Feature propagation:
      - n_src > 1:
        1. 3-NN distance-weighted interpolation from n_src → n_target points.
           - distance_reuse=True: 跳过距离计算，从 DRAM 读取已存储的距离矩阵 + sort top-3
        2. InterpN3: gather + div/mul/add interpolation engine
      - n_src <= 1:
        1. Repeat/broadcast the lone source feature to all targets (no KNN / InterpN3 cost).
      3. Concatenate with skip features (done outside; no explicit op).
      4. Shared Conv1d MLP on [B, in_channel, n_target].

    Args:
        fp_index:       Stage index (0-based), for description labels.
        n_target:       Number of upsampled (high-resolution) output points.
        n_src:          Number of source (low-resolution) input points.
        in_channel:     Input channels after concatenation (interp + skip).
        mlp:            Per-layer output channel list for the MLP.
        distance_reuse: Whether to reuse stored distance matrix from SA layers.
        src_channels:   Channels of the coarser feature tensor for InterpN3 gather.
    Returns:
        Ordered list of OperatorRef for one FP stage.
    """
    ops: List[OperatorRef] = []

    if n_src > 1:
        # Standard FP path: 3-NN interpolation (n_src -> n_target).
        k_interp = min(3, n_src)
        knn_params = {
            "input_points": n_src,
            "query_points": n_target,
            "k": k_interp,
            "distance_reuse": distance_reuse,
        }
        ops.append(OperatorRef(
            op_type="knn",
            params=knn_params,
            description=(
                f"FP{fp_index} Interpolation KNN: "
                f"query={n_target}, ref={n_src}, k={k_interp}"
                + (" [distance_reuse]" if distance_reuse else "")
            ),
        ))

        # InterpN3: gather + div/mul/add interpolation engine.
        interp_src_ch = max(1, src_channels)
        ops.append(OperatorRef(
            op_type="interp_n3",
            params={
                "num_output_points": n_target,
                "src_channels": interp_src_ch,
                "k": k_interp,
            },
            description=(
                f"FP{fp_index} InterpN3: gather+interp "
                f"{interp_src_ch}ch@{n_target}pts, k={k_interp}"
            ),
        ))

    # Shared Conv1d MLP on upsampled features
    prev_ch = in_channel
    for j, out_ch in enumerate(mlp):
        ops.append(OperatorRef(
            op_type="conv1d",
            params={
                "in_channels": prev_ch,
                "out_channels": out_ch,
                "kernel_size": 1,
                "num_points": n_target,
            },
            description=f"FP{fp_index} MLP[{j}]: Conv1d({prev_ch}→{out_ch}), {n_target} pts",
        ))
        ops.append(OperatorRef(
            op_type="bn",
            params={"num_features": out_ch, "num_points": n_target},
            description=f"FP{fp_index} MLP[{j}]: BN({out_ch})",
        ))
        ops.append(OperatorRef(
            op_type="relu",
            params={"num_features": out_ch, "num_points": n_target},
            description=f"FP{fp_index} MLP[{j}]: ReLU",
        ))
        prev_ch = out_ch

    return ops


# ---------------------------------------------------------------------------
# Helper: build segmentation head operators
# ---------------------------------------------------------------------------

def _build_seg_head_operators(
    in_channels: int,
    mid_channels: int,
    num_classes: int,
    num_points: int,
) -> List[OperatorRef]:
    """Conv1d(in→mid)+BN+ReLU + Conv1d(mid→num_classes) head."""
    return [
        OperatorRef(
            op_type="conv1d",
            params={
                "in_channels": in_channels,
                "out_channels": mid_channels,
                "kernel_size": 1,
                "num_points": num_points,
            },
            description=f"Head Conv1d({in_channels}→{mid_channels}), {num_points} pts",
        ),
        OperatorRef(
            op_type="bn",
            params={"num_features": mid_channels, "num_points": num_points},
            description=f"Head BN({mid_channels})",
        ),
        OperatorRef(
            op_type="relu",
            params={"num_features": mid_channels, "num_points": num_points},
            description="Head ReLU",
        ),
        OperatorRef(
            op_type="conv1d",
            params={
                "in_channels": mid_channels,
                "out_channels": num_classes,
                "kernel_size": 1,
                "num_points": num_points,
            },
            description=f"Head Conv1d({mid_channels}→{num_classes}), {num_points} pts",
        ),
    ]


# ===========================================================================
# PointNet2ClsSSG  —  ModelNet40 classification
# ===========================================================================

def create_pointnet2_cls_ssg(
    input_points: int = 1024,
    num_classes: int = 40,
    normal: bool = False,
) -> List[LayerDef]:
    """
    Create PointNet2ClsSSG classification network layer sequence.

    Matches model_back.py ``PointNet2ClsSSG`` (no-MSG, no-normal mode default):
        SA1: npoint=512, nsample=32,  in_ch=3,   mlp=[64,64,128]
        SA2: npoint=128, nsample=64,  in_ch=131, mlp=[128,128,256]
        SA3: group_all,  nsample=all, in_ch=259, mlp=[256,512,1024]
        FC:  1024→512→BN→ReLU → 512→256→BN→ReLU → 256→num_classes

    Args:
        input_points: Number of input points (default 1024 for ModelNet40).
        num_classes:  Number of output classes (default 40).
        normal:       Include normal vectors (adds 3 extra input channels).

    Returns:
        Ordered list of LayerDef for the full forward pass.
    """
    addition_ch = 3 if normal else 0
    layers: List[LayerDef] = []

    # ------------------------------------------------------------------
    # SA1: 1024 → 512, nsample=32
    # ------------------------------------------------------------------
    sa1_in_ch = 3 + addition_ch   # xyz (3) [+ normals (3)]
    sa1_ops = _build_sa_operators(
        sa_index=0,
        in_points=input_points,
        out_points=512,
        nsample=32,
        in_channel=sa1_in_ch,
        mlp=[64, 64, 128],
        group_all=False,
    )
    layers.append(LayerDef(
        name="SA1",
        layer_type="SA",
        operators=sa1_ops,
        description=(
            f"SA1: FPS({input_points}→512) + BallQuery(k=32) + "
            f"MLP({sa1_in_ch}→64→64→128) + MaxPool → [B,128,512]"
        ),
    ))

    # ------------------------------------------------------------------
    # SA2: 512 → 128, nsample=64
    # ------------------------------------------------------------------
    sa2_in_ch = 128 + 3           # l1_points(128) + relative_xyz(3)
    sa2_ops = _build_sa_operators(
        sa_index=1,
        in_points=512,
        out_points=128,
        nsample=64,
        in_channel=sa2_in_ch,
        mlp=[128, 128, 256],
        group_all=False,
    )
    layers.append(LayerDef(
        name="SA2",
        layer_type="SA",
        operators=sa2_ops,
        description=(
            "SA2: FPS(512→128) + BallQuery(k=64) + "
            "MLP(131→128→128→256) + MaxPool → [B,256,128]"
        ),
    ))

    # ------------------------------------------------------------------
    # SA3: GroupAll (128 → 1)
    # ------------------------------------------------------------------
    sa3_in_ch = 256 + 3           # l2_points(256) + relative_xyz(3)
    sa3_ops = _build_sa_operators(
        sa_index=2,
        in_points=128,
        out_points=1,
        nsample=128,
        in_channel=sa3_in_ch,
        mlp=[256, 512, 1024],
        group_all=True,
    )
    layers.append(LayerDef(
        name="SA3",
        layer_type="SA",
        operators=sa3_ops,
        description=(
            "SA3: GroupAll(128→1) + "
            "MLP(259→256→512→1024) + MaxPool → [B,1024,1]"
        ),
    ))

    # ------------------------------------------------------------------
    # Classifier: Linear + BN1d + ReLU × 2, then final Linear
    # ------------------------------------------------------------------
    classifier_ops = [
        OperatorRef(
            op_type="linear",
            params={"in_features": 1024, "out_features": 512, "tokens": 1},
            description="FC1: Linear(1024→512)",
        ),
        OperatorRef(
            op_type="batchnorm1d",
            params={"num_features": 512},
            description="FC1: BN1d(512)",
        ),
        OperatorRef(
            op_type="relu",
            params={"num_features": 512},
            description="FC1: ReLU",
        ),
        # Dropout(0.4) — inference no-op
        OperatorRef(
            op_type="linear",
            params={"in_features": 512, "out_features": 256, "tokens": 1},
            description="FC2: Linear(512→256)",
        ),
        OperatorRef(
            op_type="batchnorm1d",
            params={"num_features": 256},
            description="FC2: BN1d(256)",
        ),
        OperatorRef(
            op_type="relu",
            params={"num_features": 256},
            description="FC2: ReLU",
        ),
        # Dropout(0.4) — inference no-op
        OperatorRef(
            op_type="linear",
            params={"in_features": 256, "out_features": num_classes, "tokens": 1},
            description=f"FC3: Linear(256→{num_classes})",
        ),
    ]
    layers.append(LayerDef(
        name="classifier",
        layer_type="classifier",
        operators=classifier_ops,
        description=(
            f"Classifier: FC(1024→512)→BN→ReLU → "
            f"FC(512→256)→BN→ReLU → FC(256→{num_classes})"
        ),
    ))

    return layers


# ===========================================================================
# PointNet2PartSegSSG  —  ShapeNetPart part segmentation
# ===========================================================================

def create_pointnet2_partseg_ssg(
    input_points: int = 2048,
    num_classes: int = 50,
    normal: bool = False,
    enable_distance_reuse: bool = False,
) -> List[LayerDef]:
    """
    Create PointNet2PartSegSSG part-segmentation network layer sequence.

    Matches model_back.py ``PointNet2PartSegSSG`` (no-normal mode default):
        SA1: npoint=512, nsample=32,  in_ch=6,   mlp=[64,64,128]
        SA2: npoint=128, nsample=64,  in_ch=131, mlp=[128,128,256]
        SA3: group_all,               in_ch=259, mlp=[256,512,1024]
        FP1: 1→128,      in_ch=1280, mlp=[256,256]
        FP2: 128→512,    in_ch=384,  mlp=[256,128]
        FP3: 512→N,      in_ch=150,  mlp=[128,128,128]  (150=128+16+6, no-normal)
        Head: Conv1d(128→128)→BN→ReLU → Conv1d(128→num_classes)

    Args:
        input_points: Number of input points (default 2048 for ShapeNetPart).
        num_classes:  Number of part classes (default 50 for ShapeNetPart).
        normal:       Include normal vectors (adds 3 extra input channels).

    Returns:
        Ordered list of LayerDef for the full forward pass.
    """
    addition_ch = 3 if normal else 0
    layers: List[LayerDef] = []

    # ------------------------------------------------------------------
    # SA1: N → 512, nsample=32
    # in_channel = 6 + addition_ch
    # (relative_xyz=3 concatenated with l0_points=xyz=3, total 6)
    # ------------------------------------------------------------------
    sa1_in_ch = 6 + addition_ch
    sa1_ops = _build_sa_operators(
        sa_index=0,
        in_points=input_points,
        out_points=512,
        nsample=32,
        in_channel=sa1_in_ch,
        mlp=[64, 64, 128],
        group_all=False,
        dist_matrix_dram_write=enable_distance_reuse,
    )
    layers.append(LayerDef(
        name="SA1",
        layer_type="SA",
        operators=sa1_ops,
        description=(
            f"SA1: FPS({input_points}→512) + BallQuery(k=32) + "
            f"MLP({sa1_in_ch}→64→64→128) + MaxPool → [B,128,512]"
            + (" [dist_matrix_dram_write]" if enable_distance_reuse else "")
        ),
    ))

    # ------------------------------------------------------------------
    # SA2: 512 → 128, nsample=64
    # ------------------------------------------------------------------
    sa2_in_ch = 128 + 3
    sa2_ops = _build_sa_operators(
        sa_index=1,
        in_points=512,
        out_points=128,
        nsample=64,
        in_channel=sa2_in_ch,
        mlp=[128, 128, 256],
        group_all=False,
        distance_reuse=enable_distance_reuse,
    )
    layers.append(LayerDef(
        name="SA2",
        layer_type="SA",
        operators=sa2_ops,
        description=(
            "SA2: FPS(512→128) + BallQuery(k=64) + "
            "MLP(131→128→128→256) + MaxPool → [B,256,128]"
        ),
    ))

    # ------------------------------------------------------------------
    # SA3: GroupAll (128 → 1)
    # ------------------------------------------------------------------
    sa3_in_ch = 256 + 3
    sa3_ops = _build_sa_operators(
        sa_index=2,
        in_points=128,
        out_points=1,
        nsample=128,
        in_channel=sa3_in_ch,
        mlp=[256, 512, 1024],
        group_all=True,
    )
    layers.append(LayerDef(
        name="SA3",
        layer_type="SA",
        operators=sa3_ops,
        description=(
            "SA3: GroupAll(128→1) + "
            "MLP(259→256→512→1024) + MaxPool → [B,1024,1]"
        ),
    ))

    # ------------------------------------------------------------------
    # FP1 (fc1): upsample 1 → 128
    # in_channel = l2_points(256) + l3_points(1024) = 1280
    # ------------------------------------------------------------------
    fp1_ops = _build_fp_operators(
        fp_index=0,
        n_target=128,
        n_src=1,
        in_channel=1280,
        mlp=[256, 256],
        distance_reuse=False,
        src_channels=1024,
    )
    layers.append(LayerDef(
        name="FP1",
        layer_type="FP",
        operators=fp1_ops,
        description=(
            "FP1: interp(1→128) + MLP(1280→256→256) → [B,256,128]"
        ),
    ))

    # ------------------------------------------------------------------
    # FP2 (fc2): upsample 128 → 512
    # ------------------------------------------------------------------
    fp2_ops = _build_fp_operators(
        fp_index=1,
        n_target=512,
        n_src=128,
        in_channel=384,
        mlp=[256, 128],
        distance_reuse=enable_distance_reuse,
        src_channels=256,
    )
    layers.append(LayerDef(
        name="FP2",
        layer_type="FP",
        operators=fp2_ops,
        description=(
            "FP2: interp(128→512) + MLP(384→256→128) → [B,128,512]"
            + (" [distance_reuse]" if enable_distance_reuse else "")
        ),
    ))

    # ------------------------------------------------------------------
    # FP3 (fc3): upsample 512 → N
    # ------------------------------------------------------------------
    fp3_in_ch = 128 + 16 + 6 + addition_ch
    fp3_ops = _build_fp_operators(
        fp_index=2,
        n_target=input_points,
        n_src=512,
        in_channel=fp3_in_ch,
        mlp=[128, 128, 128],
        distance_reuse=enable_distance_reuse,
        src_channels=128,
    )
    layers.append(LayerDef(
        name="FP3",
        layer_type="FP",
        operators=fp3_ops,
        description=(
            f"FP3: interp(512→{input_points}) + "
            f"MLP({fp3_in_ch}→128→128→128) → [B,128,{input_points}]"
            + (" [distance_reuse]" if enable_distance_reuse else "")
        ),
    ))

    # ------------------------------------------------------------------
    # Segmentation head: Conv1d(128→128)+BN+ReLU → Conv1d(128→num_classes)
    # ------------------------------------------------------------------
    head_ops = _build_seg_head_operators(
        in_channels=128,
        mid_channels=128,
        num_classes=num_classes,
        num_points=input_points,
    )
    layers.append(LayerDef(
        name="seg_head",
        layer_type="seg_head",
        operators=head_ops,
        description=(
            f"PartSeg head: Conv1d(128→128)→BN→ReLU → "
            f"Conv1d(128→{num_classes}), {input_points} pts"
        ),
    ))

    return layers


# ===========================================================================
# PointNet2SemSSG  —  S3DIS semantic segmentation
# ===========================================================================

def create_pointnet2_sem_ssg(
    input_points: int = 4096,
    num_classes: int = 13,
    input_channels: int = 12,
    enable_distance_reuse: bool = False,
) -> List[LayerDef]:
    """
    Create PointNet2SemSSG semantic-segmentation network layer sequence.

    Matches model_back.py ``PointNet2SemSSG`` (S3DIS, 4 SA + 4 FP):
        SA1: npoint=1024, nsample=32, in_ch=12,  mlp=[32,32,64]
        SA2: npoint=256,  nsample=32, in_ch=67,  mlp=[64,64,128]
        SA3: npoint=64,   nsample=32, in_ch=131, mlp=[128,128,256]
        SA4: npoint=16,   nsample=32, in_ch=259, mlp=[256,256,512]
        FP1: 16→64,   in_ch=768, mlp=[256,256]
        FP2: 64→256,  in_ch=384, mlp=[256,256]
        FP3: 256→1024,in_ch=320, mlp=[256,128]
        FP4: 1024→N,  in_ch=128, mlp=[128,128,128]  (points1=None, no skip)
        Head: Conv1d(128→128)→BN→ReLU → Conv1d(128→num_classes)

    Args:
        input_points:   Number of input points (default 4096 for S3DIS).
        num_classes:    Number of semantic classes (default 13 for S3DIS).
        input_channels: Feature channels per point (default 12 = xyz+rgb+norm_xyz).

    Returns:
        Ordered list of LayerDef for the full forward pass.
    """
    layers: List[LayerDef] = []

    # SA channel progression: each SA in_ch = prev_out_ch + 3 (relative xyz)
    # SA1 in_ch = input_channels (as declared in model constructor)
    # SA2 in_ch = 64 + 3 = 67
    # SA3 in_ch = 128 + 3 = 131
    # SA4 in_ch = 256 + 3 = 259

    sa_configs = [
        # (sa_index, in_pts, out_pts, nsample, in_ch, mlp)
        (0, input_points, 1024, 32, input_channels,   [32, 32, 64]),
        (1, 1024,          256, 32, 64 + 3,           [64, 64, 128]),
        (2, 256,            64, 32, 128 + 3,          [128, 128, 256]),
        (3, 64,             16, 32, 256 + 3,          [256, 256, 512]),
    ]
    sa_out_pts  = [1024, 256, 64, 16]
    sa_out_ch   = [64, 128, 256, 512]

    for (sa_idx, in_pts, out_pts, nsample, in_ch, mlp) in sa_configs:
        sa_ops = _build_sa_operators(
            sa_index=sa_idx,
            in_points=in_pts,
            out_points=out_pts,
            nsample=nsample,
            in_channel=in_ch,
            mlp=mlp,
            group_all=False,
            dist_matrix_dram_write=(enable_distance_reuse and sa_idx == 0),
            distance_reuse=(enable_distance_reuse and sa_idx > 0),
        )
        _flags = []
        if enable_distance_reuse and sa_idx == 0:
            _flags.append("dist_matrix_dram_write")
        if enable_distance_reuse and sa_idx > 0:
            _flags.append("distance_reuse")
        layers.append(LayerDef(
            name=f"SA{sa_idx + 1}",
            layer_type="SA",
            operators=sa_ops,
            description=(
                f"SA{sa_idx + 1}: FPS({in_pts}→{out_pts}) + BallQuery(k={nsample}) + "
                f"MLP({in_ch}→{'→'.join(str(c) for c in mlp)}) + MaxPool "
                f"→ [B,{mlp[-1]},{out_pts}]"
                + (f" [{','.join(_flags)}]" if _flags else "")
            ),
        ))

    fp_configs = [
        # (fp_index, n_target, n_src, in_ch, mlp, src_channels)
        (0, 64,          16,   512 + 256, [256, 256], 512),
        (1, 256,         64,   256 + 128, [256, 256], 256),
        (2, 1024,        256,  256 + 64,  [256, 128], 256),
        (3, input_points, 1024, 128,      [128, 128, 128], 128),
    ]
    for (fp_idx, n_tgt, n_src, in_ch, mlp, src_ch) in fp_configs:
        fp_ops = _build_fp_operators(
            fp_index=fp_idx,
            n_target=n_tgt,
            n_src=n_src,
            in_channel=in_ch,
            mlp=mlp,
            distance_reuse=enable_distance_reuse,
            src_channels=src_ch,
        )
        layers.append(LayerDef(
            name=f"FP{fp_idx + 1}",
            layer_type="FP",
            operators=fp_ops,
            description=(
                f"FP{fp_idx + 1}: interp({n_src}→{n_tgt}) + "
                f"MLP({in_ch}→{'→'.join(str(c) for c in mlp)}) "
                f"→ [B,{mlp[-1]},{n_tgt}]"
                + (" [distance_reuse]" if enable_distance_reuse else "")
            ),
        ))

    # Segmentation head
    head_ops = _build_seg_head_operators(
        in_channels=128,
        mid_channels=128,
        num_classes=num_classes,
        num_points=input_points,
    )
    layers.append(LayerDef(
        name="sem_head",
        layer_type="sem_head",
        operators=head_ops,
        description=(
            f"SemSeg head: Conv1d(128→128)→BN→ReLU → "
            f"Conv1d(128→{num_classes}), {input_points} pts"
        ),
    ))

    return layers


# ---------------------------------------------------------------------------
# Convenience accessors
# ---------------------------------------------------------------------------

def get_pointnet2_cls_ssg_info() -> Dict[str, Any]:
    """Return info dict for PointNet2ClsSSG."""
    return {
        "name": "PointNet2ClsSSG",
        "task": "classification",
        "dataset": "ModelNet40",
        "input_shape": "[B, 3, 1024]",
        "num_classes": 40,
        "sa_stages": 3,
        "fp_stages": 0,
        "operators_used": ["fps", "knn", "conv1d", "bn", "relu",
                           "adaptive_max_pool1d", "linear", "batchnorm1d"],
    }


def get_pointnet2_partseg_ssg_info() -> Dict[str, Any]:
    """Return info dict for PointNet2PartSegSSG."""
    return {
        "name": "PointNet2PartSegSSG",
        "task": "part_segmentation",
        "dataset": "ShapeNetPart",
        "input_shape": "[B, 3, 2048]",
        "num_classes": 50,
        "sa_stages": 3,
        "fp_stages": 3,
        "operators_used": ["fps", "knn", "conv1d", "bn", "relu",
                           "adaptive_max_pool1d"],
    }


def get_pointnet2_sem_ssg_info() -> Dict[str, Any]:
    """Return info dict for PointNet2SemSSG."""
    return {
        "name": "PointNet2SemSSG",
        "task": "semantic_segmentation",
        "dataset": "S3DIS",
        "input_shape": "[B, 12, 4096]",
        "num_classes": 13,
        "sa_stages": 4,
        "fp_stages": 4,
        "operators_used": ["fps", "knn", "conv1d", "bn", "relu",
                           "adaptive_max_pool1d"],
    }


# ---------------------------------------------------------------------------
# Default instances
# ---------------------------------------------------------------------------

POINTNET2_CLS_SSG_LAYERS = create_pointnet2_cls_ssg()
"""Default PointNet2ClsSSG classification layer sequence (1024 pts, 40 classes)."""

POINTNET2_PARTSEG_SSG_LAYERS = create_pointnet2_partseg_ssg()
"""Default PointNet2PartSegSSG part-seg layer sequence (2048 pts, 50 parts)."""

POINTNET2_SEM_SSG_LAYERS = create_pointnet2_sem_ssg()
"""Default PointNet2SemSSG semantic-seg layer sequence (4096 pts, 13 classes)."""
