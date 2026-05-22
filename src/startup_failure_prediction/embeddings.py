from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_TEXT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TEXT_MODEL_ENV = "STARTUP_FAILURE_TEXT_MODEL"


def default_text_model_name() -> str:
    return os.environ.get(TEXT_MODEL_ENV, DEFAULT_TEXT_MODEL)


@lru_cache(maxsize=2)
def load_text_model(model_name: str) -> SentenceTransformer:
    try:
        return SentenceTransformer(model_name, local_files_only=True)
    except Exception as local_error:
        try:
            return SentenceTransformer(model_name)
        except Exception as remote_error:
            raise RuntimeError(
                "Could not load SentenceTransformer model "
                f"{model_name!r} from the local cache or HuggingFace. "
                "Run training once with network access, or choose a cached model."
            ) from remote_error


def encode_texts(
    texts: list[str],
    model_name: str,
    batch_size: int = 32,
) -> np.ndarray:
    model = load_text_model(model_name)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(embeddings, dtype=np.float32)
