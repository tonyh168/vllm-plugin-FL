# SPDX-License-Identifier: Apache-2.0
"""Engine-owned IndexPool pages and their uncompressed tail ring."""

from dataclasses import dataclass, replace
from math import lcm

import torch

from vllm.config import get_current_vllm_config
from vllm.model_executor.models.deepseek_v2 import DeepseekV32IndexerCache
from vllm.utils.math_utils import cdiv, next_power_of_2
from vllm.v1.attention.backend import AttentionCGSupport, AttentionMetadataBuilder
from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerBackend
from vllm.v1.kv_cache_interface import AttentionSpec

POOL_PAGE_SIZE = 64


@dataclass
class IndexPoolMetadata:
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    num_actual_tokens: int
    max_seq_len: int


class IndexPoolMetadataBuilder(AttentionMetadataBuilder[IndexPoolMetadata]):
    _cudagraph_support = AttentionCGSupport.ALWAYS

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        cm = common_attn_metadata
        return IndexPoolMetadata(
            query_start_loc=cm.query_start_loc[: cm.num_reqs + 1],
            seq_lens=cm.seq_lens[: cm.num_reqs],
            block_table=cm.block_table_tensor[: cm.num_reqs],
            num_actual_tokens=cm.num_actual_tokens,
            max_seq_len=cm.max_seq_len,
        )


class IndexPoolBackend(DeepseekV32IndexerBackend):
    @staticmethod
    def get_name():
        return "HYV4_FLASH_INDEXPOOL"

    @staticmethod
    def get_builder_cls():
        return IndexPoolMetadataBuilder

    @staticmethod
    def get_supported_kernel_block_sizes(kv_cache_spec=None):
        if kv_cache_spec is None:
            ratio = get_current_vllm_config().model_config.hf_text_config.compressed_indexer_compress_ratio
        else:
            ratio = int(kv_cache_spec.tokens_per_state)
        return [ratio * POOL_PAGE_SIZE]

    @classmethod
    def get_supported_head_sizes(cls):
        return [132]


class IndexPoolTailBackend(IndexPoolBackend):
    @staticmethod
    def get_name():
        return "HYV4_FLASH_INDEXPOOL_TAIL"


@dataclass(frozen=True, kw_only=True)
class FlashCircularBufferSpec(AttentionSpec):
    """One block per request for the uncompressed tail of an open group."""
    num_kv_heads: int = 1
    head_size: int = 131
    dtype: torch.dtype = torch.float32

    @property
    def block_table_token_alignment(self):
        return None

    def max_memory_usage_bytes(self, vllm_config):
        return self.page_size_bytes

    def max_num_blocks_per_req(self, vllm_config, max_len):
        return 1

    def is_uniform_with_collection(self, kv_cache_specs):
        return all(isinstance(spec, FlashCircularBufferSpec) for spec in kv_cache_specs.values())

    @property
    def prefix_cacheable(self):
        return False

    @property
    def uses_slot_mapping(self):
        return False


def tail_ring_size(ratio, speculative_tokens):
    return ratio * next_power_of_2(cdiv(ratio + speculative_tokens, ratio))


class IndexPoolCache(DeepseekV32IndexerCache):
    def __init__(self, prefix, cache_config, ratio, *, tail=False):
        super().__init__(head_dim=131 if tail else 132,
                         dtype=torch.float32 if tail else torch.uint8,
                         prefix=prefix, cache_config=cache_config)
        self.ratio = ratio
        self.tail = tail

    def get_attn_backend(self):
        return IndexPoolTailBackend if self.tail else IndexPoolBackend

    def get_kv_cache_spec(self, vllm_config):
        ring = tail_ring_size(self.ratio, vllm_config.num_speculative_tokens)
        if self.tail:
            return FlashCircularBufferSpec(block_size=ring)
        spec = super().get_kv_cache_spec(vllm_config)
        assert isinstance(spec, AttentionSpec)
        # Older 0.24 AttentionSpec lacks tokens_per_state; attach it without
        # changing the global class, keeping this extension model-local.
        object.__setattr__(spec, "tokens_per_state", self.ratio)
        return replace(spec, block_size=lcm(self.cache_config.block_size,
                                            self.ratio * POOL_PAGE_SIZE, ring))

    def bind_kv_cache(self, kv_cache):
        super().bind_kv_cache(kv_cache)
        if not self.tail and self.kv_cache.shape[1] != POOL_PAGE_SIZE:
            raise ValueError("IndexPool must be bound as 64-state kernel pages")
