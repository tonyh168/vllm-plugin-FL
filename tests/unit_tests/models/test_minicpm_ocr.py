# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from vllm_fl.models.minicpm_ocr.registration import register_minicpm_ocr_support


def test_registration_is_idempotent_and_preserves_existing_models():
    from transformers import AutoConfig

    from vllm.model_executor.models.registry import ModelRegistry

    previous = set(ModelRegistry.get_supported_archs())
    assert register_minicpm_ocr_support() is True
    assert register_minicpm_ocr_support() is True
    assert previous <= set(ModelRegistry.get_supported_archs())
    config = AutoConfig.for_model("minicpm_ocr")
    assert config.downsample_mode == "4x"
    assert config.drop_vision_last_layer is False
    assert config.tie_word_embeddings is True


def test_actual_processor_registration():
    from transformers.models.auto.processing_auto import processor_class_from_name

    from vllm_fl.models.minicpm_ocr.processor import MiniCPMOcrProcessor

    register_minicpm_ocr_support()
    assert processor_class_from_name("MiniCPMOcrProcessor") is MiniCPMOcrProcessor


@pytest.mark.parametrize(
    "value,count,expected",
    [
        (None, 2, [None, None]),
        (True, 2, [True, True]),
        ([True, False], 2, [True, False]),
        (torch.tensor([False, True]), 2, [False, True]),
    ],
)
def test_merger_flags(value, count, expected):
    from vllm_fl.models.minicpm_ocr.model import _expand_flags

    assert _expand_flags(value, count) == expected


def test_merger_flags_reject_wrong_coverage():
    from vllm_fl.models.minicpm_ocr.model import _expand_flags

    with pytest.raises(ValueError):
        _expand_flags([True, False], 3)


def test_video_processor_registration():
    from transformers.models.auto.video_processing_auto import VIDEO_PROCESSOR_MAPPING

    from vllm_fl.models.minicpm_ocr.config import MiniCPMOcrConfig
    from vllm_fl.models.minicpm_ocr.video_processing_minicpm_ocr import (
        MiniCPMOcrVideoProcessor,
    )

    register_minicpm_ocr_support()
    assert VIDEO_PROCESSOR_MAPPING[MiniCPMOcrConfig] is MiniCPMOcrVideoProcessor


@pytest.mark.parametrize("mode,alignment", [("4x", 2), ("16x", 4)])
def test_resize_override_is_per_call(mode, alignment):
    from vllm_fl.models.minicpm_ocr.image_processing_minicpm_ocr import (
        MiniCPMOcrImageProcessor,
    )

    processor = MiniCPMOcrImageProcessor()
    image = torch.zeros((3, 73, 123), dtype=torch.uint8)
    result = processor(
        image,
        downsample_mode=mode,
        disable_small_image_upsize=True,
        return_tensors="pt",
    )
    assert processor.downsample_mode == "4x"
    assert processor.disable_small_image_upsize is False
    assert (result["target_sizes"] % alignment == 0).all()
    assert (
        result["pixel_values"].shape[-1]
        == int(result["target_sizes"].prod(-1).sum()) * processor.patch_size
    )
    default = processor(image, return_tensors="pt")
    assert (
        default["target_sizes"].prod(-1).sum() > result["target_sizes"].prod(-1).sum()
    )


def test_ocr_grid_uses_rows_then_columns_for_non_square_image():
    from vllm_fl.models.minicpm_ocr.image_processing_minicpm_ocr import (
        MiniCPMOcrImageProcessor,
    )
    from vllm_fl.models.minicpm_ocr.video_processing_minicpm_ocr import (
        MiniCPMOcrVideoProcessor,
    )

    image = MiniCPMOcrImageProcessor()
    video = MiniCPMOcrVideoProcessor()
    assert image.get_sliced_grid((896, 1344), 9, 448) == [2, 3]
    assert video.get_sliced_grid((896, 1344), 9, 448) == [2, 3]
    assert image.get_sliced_grid((1344, 896), 9, 448) == [3, 2]


def test_modality_options_are_resolved_without_image_resize_leaking_to_video():
    from types import SimpleNamespace

    from vllm_fl.models.minicpm_ocr.image_processing_minicpm_ocr import (
        MiniCPMOcrImageProcessor,
    )
    from vllm_fl.models.minicpm_ocr.model import (
        MiniCPMOcrMultiModalProcessor,
        MiniCPMOcrProcessingInfo,
    )

    # This is a routing test; processor numerics are exercised independently.
    info = object.__new__(MiniCPMOcrProcessingInfo)
    info.ctx = SimpleNamespace(get_merged_mm_kwargs=lambda kwargs: kwargs)
    info.get_image_processor = lambda: MiniCPMOcrImageProcessor()
    processor = object.__new__(MiniCPMOcrMultiModalProcessor)
    processor.info = info
    kwargs = {
        "adaptive_small_image_upsize": True,
        "images_kwargs": {"downsample_mode": "16x", "max_slice_nums": 3},
        "videos_kwargs": {"downsample_mode": "4x"},
    }
    images = processor._processor_kwargs(kwargs, "image")
    videos = processor._processor_kwargs(kwargs, "video")
    assert images["downsample_mode"] == "16x"
    assert images["max_slice_nums"] == 3
    assert images["adaptive_small_image_upsize"] is True
    assert videos["downsample_mode"] == "4x"
    assert "adaptive_small_image_upsize" not in videos
    assert kwargs["adaptive_small_image_upsize"] is True


@pytest.mark.parametrize("mode", ["4x", "16x"])
@pytest.mark.parametrize("tensor_type", [None, "pt", "np"])
def test_adaptive_batch_preserves_per_image_content(mode, tensor_type):
    import numpy as np
    from PIL import Image, ImageDraw

    from vllm_fl.models.minicpm_ocr.image_processing_minicpm_ocr import (
        MiniCPMOcrImageProcessor,
    )

    blank = Image.new("RGB", (83, 47), "white")
    dense = blank.copy()
    draw = ImageDraw.Draw(dense)
    for y in range(2, 43, 6):
        for x in range(2, 79, 5):
            draw.rectangle((x, y, x + 1, y + 3), fill="black")
    processor = MiniCPMOcrImageProcessor()
    kwargs = dict(
        downsample_mode=mode,
        adaptive_small_image_upsize=True,
        return_tensors=tensor_type,
    )
    singles = [processor(image, **kwargs) for image in (blank, dense)]
    batch = processor([blank, dense], **kwargs)
    # Same shape, different pixels: tiny blank -> 84, dense small glyphs -> 448.
    assert int(singles[0]["target_sizes"].prod()) < int(
        singles[1]["target_sizes"].prod()
    )
    for key, dim in [("pixel_values", -1), ("target_sizes", 0)]:
        values = [s[key] for s in singles]
        if isinstance(batch[key], torch.Tensor):
            torch.testing.assert_close(
                batch[key], torch.cat(values, dim=dim), rtol=0, atol=0
            )
        else:
            np.testing.assert_array_equal(batch[key], np.concatenate(values, axis=dim))
    assert batch["grids"] == [grid for s in singles for grid in s["grids"]]
    assert batch["num_patches_per_image"] == [
        n for s in singles for n in s["num_patches_per_image"]
    ]
    assert not hasattr(processor, "_source_resize")
    assert processor.downsample_mode == "4x"
    assert processor.adaptive_small_image_upsize is False


def test_parallel_resize_options_do_not_leak_to_shared_processor():
    from concurrent.futures import ThreadPoolExecutor

    from vllm_fl.models.minicpm_ocr.image_processing_minicpm_ocr import (
        MiniCPMOcrImageProcessor,
    )

    processor = MiniCPMOcrImageProcessor()
    image = torch.full((3, 47, 83), 255, dtype=torch.uint8)
    options = [
        dict(downsample_mode=mode, **resize)
        for mode in ["4x", "16x"]
        for resize in [
            dict(disable_small_image_upsize=True),
            dict(adaptive_small_image_upsize=True),
        ]
    ]

    def call(kwargs):
        return processor(image, return_tensors="pt", **kwargs)

    expected = [call(kwargs) for kwargs in options]
    with ThreadPoolExecutor(max_workers=4) as pool:
        actual = list(pool.map(call, options * 2))
    for i, result in enumerate(actual):
        for key in ["pixel_values", "target_sizes"]:
            torch.testing.assert_close(
                result[key], expected[i % 4][key], rtol=0, atol=0
            )
    assert not hasattr(processor, "_source_resize")
    assert not processor.adaptive_small_image_upsize
    assert not processor.disable_small_image_upsize


def test_missing_transformers_base_preserves_other_model_registrations(monkeypatch):
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES

    from vllm.model_executor.models.registry import ModelRegistry
    from vllm.transformers_utils.config import _CONFIG_REGISTRY

    previous_models = set(ModelRegistry.get_supported_archs())
    previous_configs = dict(_CONFIG_REGISTRY)
    monkeypatch.delitem(CONFIG_MAPPING_NAMES, "minicpmv4_6")
    assert register_minicpm_ocr_support() is False
    assert set(ModelRegistry.get_supported_archs()) == previous_models
    assert previous_configs == _CONFIG_REGISTRY
