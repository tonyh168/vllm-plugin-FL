# GLM5-Next kernel provenance

The kpool implementation is vendored from the supplied
`glm-5-next-adapt-0808` vLLM reference tree:

- reference commit: `a9b39d6` plus `vllm-glm5next-0808.patch`
- original paths:
  - `vllm/models/glm5next/nvidia/ops/kpool_compress.py`
  - `vllm/model_executor/layers/sparse_attn_indexer_kpool.py`
- license: Apache-2.0

Before vendoring, the equivalent Hugging Face Transformers and SGLang
implementations in the same handoff bundle were checked. The plugin keeps the
reference formulas and cache lifecycle; only import paths and vLLM 0.24
metadata compatibility reads differ.

The upstream vLLM repository was re-checked at commit
`bd6536071cec4dcd8cf91c0e2aa04aec83fc1c37` (2026-08-10). Its Kimi K3 KDA,
KDA metadata, and latent-MoE tail files are byte-identical to the copies in the
supplied `a9b39d6` reference tree. Kimi K3 confirms the bounded-gate wrapper and
`eager_break_during_capture` pattern; its latent-MoE tail is not a KV tail cache
and was not transplanted into the GLM kpool implementation.
