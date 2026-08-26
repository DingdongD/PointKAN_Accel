"""
与 FractalCloud/cfgs 中 YAML 对齐的 encoder 预设（可直接 merge 进训练配置）。

PointVector：S / L / XL 对应 modelnet40、s3dis 等官方风格配置。
PointNeXt：S（ModelNet）、L（S3DIS）等。
"""


def pointnext_encoder_modelnet40_s(**overrides):
    """cfgs/modelnet40ply2048/pointnext-s.yaml encoder_args"""
    cfg = {
        "NAME": "PointNextEncoder",
        "stage": [0, 1, 2, 3, 4, 5],
        "blocks": [1, 1, 1, 1, 1, 1],
        "strides": [1, 2, 2, 2, 2, 1],
        "width": 32,
        "in_channels": 3,
        "radius": 0.15,
        "radius_scaling": 1.5,
        "sa_layers": 2,
        "sa_use_res": True,
        "nsample": 32,
        "expansion": 4,
        "aggr_args": {"feature_type": "dp_fj", "reduction": "max"},
        "group_args": {"NAME": "ballquery", "normalize_dp": True},
        "conv_args": {"order": "conv-norm-act"},
        "act_args": {"act": "relu"},
        "norm_args": {"norm": "bn"},
    }
    cfg.update(overrides)
    return cfg


def pointnext_encoder_s3dis_l(**overrides):
    """cfgs/s3dis/pointnext-l.yaml encoder_args"""
    cfg = {
        "NAME": "PointNextEncoder",
        "blocks": [1, 3, 5, 3, 3],
        "strides": [1, 4, 4, 4, 4],
        "sa_layers": 1,
        "sa_use_res": False,
        "width": 32,
        "in_channels": 4,
        "expansion": 4,
        "radius": 0.1,
        "nsample": 32,
        "aggr_args": {"feature_type": "dp_fj", "reduction": "max"},
        "group_args": {"NAME": "ballquery", "normalize_dp": True},
        "conv_args": {"order": "conv-norm-act"},
        "act_args": {"act": "relu"},
        "norm_args": {"norm": "bn"},
    }
    cfg.update(overrides)
    return cfg


def pointvector_encoder_modelnet40_s(**overrides):
    """cfgs/modelnet40ply2048/pointvector-s.yaml encoder_args"""
    cfg = {
        "NAME": "PointVectorEncoder",
        "blocks": [1, 1, 1, 1, 1, 1],
        "strides": [1, 2, 2, 2, 2, 1],
        "width": 32,
        "in_channels": 3,
        "radius": 0.15,
        "radius_scaling": 1.5,
        "sa_layers": 2,
        "sa_use_res": True,
        "nsample": 32,
        "expansion": 4,
        "flag": 0,
        "aggr_args": {"feature_type": "dp_fj", "reduction": "max"},
        "group_args": {"NAME": "ballquery", "normalize_dp": True},
        "conv_args": {"order": "conv-norm-act"},
        "act_args": {"act": "relu"},
        "norm_args": {"norm": "bn"},
    }
    cfg.update(overrides)
    return cfg


def pointvector_encoder_s3dis_l(**overrides):
    """cfgs/s3dis/pointvector-l.yaml"""
    cfg = {
        "NAME": "PointVectorEncoder",
        "blocks": [1, 3, 5, 3, 3],
        "strides": [1, 4, 4, 4, 4],
        "sa_layers": 1,
        "sa_use_res": False,
        "width": 32,
        "in_channels": 4,
        "expansion": 4,
        "radius": 0.1,
        "nsample": 32,
        "flag": 1,
        "aggr_args": {"feature_type": "dp_fj", "reduction": "max"},
        "group_args": {"NAME": "ballquery", "normalize_dp": True},
        "conv_args": {"order": "conv-norm-act"},
        "act_args": {"act": "relu"},
        "norm_args": {"norm": "bn"},
    }
    cfg.update(overrides)
    return cfg


def pointvector_encoder_s3dis_xl(**overrides):
    """cfgs/s3dis/pointvector-xl.yaml（更宽）"""
    cfg = pointvector_encoder_s3dis_l()
    cfg["width"] = 64
    cfg["blocks"] = [1, 4, 7, 4, 4]
    cfg.update(overrides)
    return cfg


def pointnext_encoder_shapenetpart_s(**overrides):
    """与 pointvector_encoder_shapenetpart_s 同结构，骨干为 PointNextEncoder。"""
    cfg = pointvector_encoder_shapenetpart_s()
    cfg["NAME"] = "PointNextEncoder"
    cfg.update(overrides)
    return cfg


def pointvector_encoder_shapenetpart_s(**overrides):
    """cfgs/shapenetpart/pointvector-s_c64.yaml encoder_args"""
    cfg = {
        "NAME": "PointVectorEncoder",
        "blocks": [1, 1, 1, 1, 1],
        "strides": [1, 2, 2, 2, 2],
        "width": 64,
        "in_channels": 7,
        "sa_layers": 2,
        "sa_use_res": True,
        "radius": 0.1,
        "radius_scaling": 2.5,
        "nsample": 32,
        "expansion": 4,
        "flag": 2,
        "aggr_args": {"feature_type": "dp_fj", "reduction": "max"},
        "group_args": {"NAME": "ballquery", "normalize_dp": True},
        "conv_args": {"order": "conv-norm-act"},
        "act_args": {"act": "relu"},
        "norm_args": {"norm": "bn"},
    }
    cfg.update(overrides)
    return cfg
