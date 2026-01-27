#!/usr/bin/env python
# -*- coding: utf-8 -*-

import mteb
import numpy as np
import torch
import torch.nn.functional as F


# Dimensionality of the random Gaussian embeddings
EMBEDDING_DIM = 256


class RandomGaussianModel:
    """
    Simple baseline encoder that ignores the input text and returns
    random Gaussian embeddings.

    Each call to `encode` produces embeddings sampled from N(0, 1),
    optionally L2-normalized. This can be used as a sanity-check
    / lower-bound baseline for MTEB evaluations.
    """

    def __init__(self, embedding_dim=EMBEDDING_DIM, normalize_embeddings=True, seed=42):
        self.embedding_dim = embedding_dim
        self.normalize_embeddings = normalize_embeddings

        # Use a torch Generator for reproducibility across runs
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)

    @torch.inference_mode()
    def encode(
        self,
        sentences,
        batch_size=1,
        convert_to_numpy=True,
        show_progress_bar=False,
        **_,
    ):
        """
        MTEB-compatible encode function.

        Parameters
        ----------
        sentences : Union[str, List[str]]
            Input sentences (ignored; only length is used).
        batch_size : int
            Batch size for generating embeddings.
        convert_to_numpy : bool
            If True, returns a numpy array; otherwise returns a torch.Tensor.
        show_progress_bar : bool
            If True, show a progress bar over batches.

        Returns
        -------
        np.ndarray or torch.Tensor
            Array / tensor of shape (N, embedding_dim) with random Gaussian values.
        """
        if isinstance(sentences, str):
            sentences = [sentences]

        outs = []

        indices = range(0, len(sentences), batch_size)
        if show_progress_bar:
            import tqdm

            indices = tqdm.tqdm(indices)

        for i in indices:
            batch = sentences[i : i + batch_size]
            bsz = len(batch)

            # Sample from N(0, 1)
            embs = torch.randn(
                (bsz, self.embedding_dim),
                generator=self.generator,
                dtype=torch.float32,
            )

            if self.normalize_embeddings:
                embs = F.normalize(embs, p=2, dim=-1)

            if convert_to_numpy:
                outs.append(embs.cpu().numpy())
            else:
                outs.append(embs)

        if convert_to_numpy:
            return np.concatenate(outs, axis=0)

        return torch.cat(outs, dim=0)


def main():
    # Instantiate the random baseline model
    model = RandomGaussianModel(
        embedding_dim=EMBEDDING_DIM,
        normalize_embeddings=True,
        seed=42,
    )

    # Choose the same tasks as in your original script
    tasks = mteb.get_tasks(
        tasks=[
            "AmazonCounterfactualClassification",
            "DBpediaClassification",
            "TweetTopicSingleClassification",
            "FinancialPhrasebankClassification",
        ]
    )

    evaluation = mteb.MTEB(tasks=tasks)
    evaluation.run(model)


if __name__ == "__main__":
    main()
