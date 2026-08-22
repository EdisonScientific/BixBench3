from __future__ import annotations

from inspect_ai.model import ModelInfo, set_model_info


def register_model_info() -> None:
    set_model_info(
        "z-ai/glm-5.2",
        ModelInfo(
            organization="Z.ai",
            model="GLM 5.2",
            context_length=1_048_576,
            reasoning=True,
        ),
    )
