from openpoints_mini.utils.registry import Registry

MODELS = Registry("models")


def build_model_from_cfg(cfg, **kwargs):
    """Build registered model from config dict with key NAME (same as OpenPoints)."""
    return MODELS.build(cfg, **kwargs)
