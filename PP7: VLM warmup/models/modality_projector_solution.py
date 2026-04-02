"""A minimal modality projector for mapping vision features into LM space."""

import torch.nn as nn


class ModalityProjector(nn.Module):
    """Project each visual token embedding into the language-model hidden size."""

    def __init__(self, vision_hidden_dim, language_hidden_dim):
        """Initialize the modality projector.

        Args:
            vision_hidden_dim: Hidden size produced by the vision backbone.
            language_hidden_dim: Hidden size expected by the language model.
        """
        super().__init__()
        self.vision_hidden_dim = vision_hidden_dim
        self.language_hidden_dim = language_hidden_dim
        self.proj = nn.Linear(vision_hidden_dim, language_hidden_dim, bias=False)

    def forward(self, x):
        """Apply the learned linear map to every visual token independently.

        Args:
            x: Visual token embeddings of shape `[batch, num_tokens, d_vision]`.

        Returns:
            Projected token embeddings of shape `[batch, num_tokens, d_language]`.
        """
        return self.proj(x)
