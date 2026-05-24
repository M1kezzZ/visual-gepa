"""Visual-GEPA: Failure-Clustered Visual Reflection for prompt evolution on long-horizon computer-use agents."""

__version__ = "0.0.1"

from .patch_schema import FCVRPatch
from .structured_prompt import StructuredPrompt

__all__ = ["FCVRPatch", "StructuredPrompt", "__version__"]
