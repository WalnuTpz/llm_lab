"""Text generation utilities."""

from llm_lab.generation.generate import generate
from llm_lab.generation.sampler import top_p_sample

__all__ = ["generate", "top_p_sample"]
