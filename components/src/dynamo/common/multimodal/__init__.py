# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Multimodal utilities for Dynamo components.

Torch-backed symbols are resolved lazily so that torch-free submodules
(e.g. nvdec_decoder) can be imported in images that do not ship torch.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING

from dynamo.common.constants import EmbeddingTransferMode
from dynamo.common.multimodal.audio_loader import AudioLoader
from dynamo.common.multimodal.image_loader import ImageLoader
from dynamo.common.multimodal.video_loader import VideoLoader

if TYPE_CHECKING:
    from dynamo.common.multimodal.async_encoder_cache import AsyncEncoderCache
    from dynamo.common.multimodal.embedding_transfer import (
        AbstractEmbeddingReceiver,
        AbstractEmbeddingSender,
        LocalEmbeddingReceiver,
        LocalEmbeddingSender,
        NixlReadEmbeddingReceiver,
        NixlReadEmbeddingSender,
        NixlWriteEmbeddingReceiver,
        NixlWriteEmbeddingSender,
        TransferRequest,
    )

_LAZY = {
    "AsyncEncoderCache": "dynamo.common.multimodal.async_encoder_cache",
    "AbstractEmbeddingReceiver": "dynamo.common.multimodal.embedding_transfer",
    "AbstractEmbeddingSender": "dynamo.common.multimodal.embedding_transfer",
    "LocalEmbeddingReceiver": "dynamo.common.multimodal.embedding_transfer",
    "LocalEmbeddingSender": "dynamo.common.multimodal.embedding_transfer",
    "NixlReadEmbeddingReceiver": "dynamo.common.multimodal.embedding_transfer",
    "NixlReadEmbeddingSender": "dynamo.common.multimodal.embedding_transfer",
    "NixlWriteEmbeddingReceiver": "dynamo.common.multimodal.embedding_transfer",
    "NixlWriteEmbeddingSender": "dynamo.common.multimodal.embedding_transfer",
    "TransferRequest": "dynamo.common.multimodal.embedding_transfer",
}


def __getattr__(name: str):
    if name in _LAZY:
        import importlib

        return getattr(importlib.import_module(_LAZY[name]), name)
    if name in ("EMBEDDING_SENDER_FACTORIES", "EMBEDDING_RECEIVER_FACTORIES"):
        return _factories()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _factories() -> dict[str, dict[EmbeddingTransferMode, Callable]]:
    from dynamo.common.multimodal import embedding_transfer as et

    return {
        "EMBEDDING_SENDER_FACTORIES": {
            EmbeddingTransferMode.LOCAL: et.LocalEmbeddingSender,
            EmbeddingTransferMode.NIXL_WRITE: et.NixlWriteEmbeddingSender,
            EmbeddingTransferMode.NIXL_READ: et.NixlReadEmbeddingSender,
        },
        "EMBEDDING_RECEIVER_FACTORIES": {
            EmbeddingTransferMode.LOCAL: et.LocalEmbeddingReceiver,
            EmbeddingTransferMode.NIXL_WRITE: et.NixlWriteEmbeddingReceiver,
            # [gluo FIXME] can't use pre-registered tensor as NIXL requires descriptors
            # to be at matching size, need to overwrite nixl connect library
            EmbeddingTransferMode.NIXL_READ: lambda: et.NixlReadEmbeddingReceiver(
                max_items=0
            ),
        },
    }


__all__ = [
    "AsyncEncoderCache",
    "AudioLoader",
    "EMBEDDING_RECEIVER_FACTORIES",
    "EMBEDDING_SENDER_FACTORIES",
    "ImageLoader",
    "VideoLoader",
    "NixlReadEmbeddingReceiver",
    "NixlReadEmbeddingSender",
    "NixlWriteEmbeddingSender",
    "NixlWriteEmbeddingReceiver",
    "TransferRequest",
    "LocalEmbeddingReceiver",
    "LocalEmbeddingSender",
]
