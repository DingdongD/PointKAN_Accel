# Register all MODELS (import side effects)
from openpoints_mini.models import backbone  # noqa: F401
from openpoints_mini.models.classification import base_cls  # noqa: F401
from openpoints_mini.models.segmentation import base_seg  # noqa: F401
