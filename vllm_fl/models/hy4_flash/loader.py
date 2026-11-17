# SPDX-License-Identifier: Apache-2.0
"""Expert-sliced loader for Flash, including DFly/MTP exclusion."""

from vllm_fl.models.hy4.loader import HYV4SafetensorsLoader


class HYV4FlashSafetensorsLoader(HYV4SafetensorsLoader):
    def load_weights(self, model, model_config):
        self._init_ep_weight_filter(model_config)
        if self.local_expert_ids is None:
            raise ValueError("Flash requires --enable-expert-parallel --enable-ep-weight-filter")
        target = getattr(model, "language_model", model)
        target._hy4_local_expert_ids = sorted(self.local_expert_ids)
        try:
            # Bypass the parent class-name check (which knows only HYV4).
            from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
            DefaultModelLoader.load_weights(self, model, model_config)
        finally:
            del target._hy4_local_expert_ids

    def _get_weights_iterator(self, source):
        for name, value in super()._get_weights_iterator(source):
            if name.startswith(("model.dfly_layers.", "model.mtp_layers.")):
                continue
            yield name, value
