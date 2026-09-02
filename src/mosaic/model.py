"""Independent additive MOSAIC baseline used by P4/S0 and A0 smoke."""

from __future__ import annotations

from typing import Dict, Mapping, Optional

import torch
from torch import nn


BIO_CATEGORICAL = ("Medium", "Temperature")
MEASUREMENT_CATEGORICAL = ("data_source", "instrument", "Yeast_cell_plate")


class AdditiveMosaic(nn.Module):
    """Biological and measurement paths with no strain/compound ID embedding."""

    def __init__(
        self,
        n_proteins: int,
        cardinalities: Mapping[str, int],
        strain_external_dim: int,
        compound_external_dim: int,
        hidden_dim: int = 32,
        embedding_dim: int = 8,
        dropout: float = 0.0,
        memorization_samples: int = 0,
    ) -> None:
        super().__init__()
        self.n_proteins = int(n_proteins)
        self.hidden_dim = int(hidden_dim)
        self.bio_embeddings = nn.ModuleDict(
            {
                key: nn.Embedding(int(cardinalities[key]), embedding_dim, padding_idx=0)
                for key in BIO_CATEGORICAL
            }
        )
        self.measurement_embeddings = nn.ModuleDict(
            {
                key: nn.Embedding(int(cardinalities[key]), embedding_dim, padding_idx=0)
                for key in MEASUREMENT_CATEGORICAL
            }
        )
        self.bio_category_projection = nn.Linear(embedding_dim, hidden_dim, bias=False)
        self.measurement_category_projection = nn.Linear(embedding_dim, hidden_dim, bias=False)
        self.time_projection = nn.Linear(2, hidden_dim, bias=False)
        self.strain_projection = (
            nn.Linear(strain_external_dim, hidden_dim, bias=False)
            if strain_external_dim > 0
            else None
        )
        self.compound_projection = (
            nn.Linear(compound_external_dim, hidden_dim, bias=False)
            if compound_external_dim > 0
            else None
        )
        self.dropout = nn.Dropout(dropout)
        self.bio_decoder = nn.Linear(hidden_dim, self.n_proteins, bias=False)
        self.measurement_decoder = nn.Linear(hidden_dim, self.n_proteins, bias=False)
        self.output_bias = nn.Parameter(torch.zeros(self.n_proteins))
        self.memorization = (
            nn.Embedding(memorization_samples, self.n_proteins)
            if memorization_samples > 0
            else None
        )
        if self.memorization is not None:
            nn.init.zeros_(self.memorization.weight)

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        smoke_sample_index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch = inputs["time_features"].shape[0]
        device = inputs["time_features"].device
        dtype = inputs["time_features"].dtype
        biological = self.time_projection(inputs["time_features"])
        for key, embedding in self.bio_embeddings.items():
            biological = biological + self.bio_category_projection(embedding(inputs[key]))
        measurement = torch.zeros((batch, self.hidden_dim), device=device, dtype=dtype)
        for key, embedding in self.measurement_embeddings.items():
            measurement = measurement + self.measurement_category_projection(embedding(inputs[key]))

        if self.strain_projection is not None:
            values = inputs.get("strain_external")
            mask = inputs.get("strain_external_mask")
            if values is not None and mask is not None:
                biological = biological + self.strain_projection(values) * mask[:, None]
        if self.compound_projection is not None:
            values = inputs.get("compound_external")
            mask = inputs.get("compound_external_mask")
            if values is not None and mask is not None:
                biological = biological + self.compound_projection(values) * mask[:, None]
        prediction = self.output_bias + self.bio_decoder(self.dropout(torch.tanh(biological)))
        prediction = prediction + self.measurement_decoder(self.dropout(torch.tanh(measurement)))
        if self.memorization is not None:
            if smoke_sample_index is None:
                raise ValueError("smoke_sample_index is required by the overfit-only memorization head")
            prediction = prediction + self.memorization(smoke_sample_index)
        return prediction

    def measurement_penalty(self) -> torch.Tensor:
        penalty = self.measurement_decoder.weight.square().mean()
        for embedding in self.measurement_embeddings.values():
            penalty = penalty + embedding.weight.square().mean()
        return penalty

