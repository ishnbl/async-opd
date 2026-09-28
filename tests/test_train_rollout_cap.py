"""Training-only response limits for OPD and OPSD."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from opd.coordinator.opd_mode import OPDMode, OPSDMode
from opd.coordinator.fused_hybrid_sync import FusedHybridOPDMode
from opd.trainer.base import BaseBackend
from opd.utils.config import DataConfig, OPDConfig, RolloutConfig


def test_train_cap_yaml_override_and_validation():
    path = Path(__file__).resolve().parents[1] / "configs/examples/opsd_gsm8k_0.5b_2gpu.yaml"
    cfg = OPDConfig.from_yaml(path, overrides=["rollout.train_max_tokens=128"])
    assert cfg.rollout.train_max_tokens == 128
    for bad in (0, -1, 513):
        cfg.rollout.train_max_tokens = bad
        with pytest.raises(ValueError, match="train_max_tokens"):
            cfg.validate()


@pytest.mark.parametrize("mode_cls", [OPDMode, OPSDMode])
def test_train_cap_only_applies_to_training_generation(mode_cls):
    submitted = []

    class Proxy:
        def submit_generate(self, batch):
            submitted.append(dict(batch))

    cfg = OPDConfig(rollout=RolloutConfig(train_max_tokens=2))
    mode = mode_cls(rollout_proxy=Proxy(), teacher_client=None, trainer_proxy=None,
                    tracer=None, opd_config=cfg)
    mode.async_generate({"input_ids": torch.tensor([[1]]), "solutions": ["a"]})
    mode.async_generate({"input_ids": torch.tensor([[1]]), "eval": True,
                         "max_response_length": 7})
    assert submitted[0]["max_response_length"] == 2
    assert submitted[1]["max_response_length"] == 7


def test_trainer_masks_only_short_generated_response():
    backend = SimpleNamespace(rank=0, dp_rank=0, world_size=1, dp_world_size=1,
                              mini_batch_size=1, max_response_length=4)
    batch = {
        "input_ids": torch.tensor([[0, 11, 12, 21, 22]]),
        "attention_mask": torch.tensor([[0, 1, 1, 1, 1]]),
        "responses": torch.tensor([[21, 22]]),
        "prompt_lengths": torch.tensor([2]),
    }
    prepared = BaseBackend._prepare_train_batch(backend, batch)
    assert prepared["max_prompt"] == 3
    assert prepared["response_mask"].tolist() == [[False, False, False, True, True]]


def test_fused_training_cap_does_not_limit_evaluation():
    submitted = []

    class TrainerProxy:
        def submit_command_async(self, command, payload):
            submitted.append(payload["options"]["max_response_length"])

    cfg = OPDConfig(data=DataConfig(max_response_length=8),
                    rollout=RolloutConfig(train_max_tokens=2))
    mode = FusedHybridOPDMode(rollout_proxy=None, teacher_client=None,
                              trainer_proxy=TrainerProxy(), tracer=None, opd_config=cfg)
    mode.async_generate({"input_ids": torch.tensor([[1]])})
    mode.async_generate({"input_ids": torch.tensor([[1]]), "eval": True,
                         "max_response_length": 7})
    assert submitted == [2, 7]
