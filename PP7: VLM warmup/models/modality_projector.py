"""Exercise stub for the PP7 modality projector."""

import torch.nn as nn


class ModalityProjector(nn.Module):
    """Student implementation target for the modality projector exercise."""

    def __init__(self, vision_dim, llm_dim):
        """Placeholder initializer for the exercise implementation.

        Args:
            *args: Positional arguments the student-defined projector may need.
            **kwargs: Keyword arguments the student-defined projector may need.
        """
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(vision_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim))

    # TODO implement the modality projector.
    def forward(self, x):
         
        # TODO implement the forward pass of the modality projector.
        return self.projector(x)

