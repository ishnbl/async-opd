"""Coordinator selection for bounded off-policy self-distillation."""

from pathlib import Path
from queue import Queue

import pytest
import torch

from opd.coordinator.factory import create_coordinator
from opd.coordinator.opd_mode import OPSDMode
from opd.coordinator.step_off import StepOffCoordinator
from opd.data.prompt import format_prompt
from opd.loss.kl import KLConfig
from opd.trainer.opd import OPDTrainer
from opd.utils.config import (
    AlgorithmConfig,
    DataConfig,
    ModelConfig,
    NStepOffConfig,
    OPDConfig,
    PipelineConfig,
    RolloutConfig,
    TrainerConfig,
)


def test_opsd_accepts_one_queued_rollout_on_separate_rollout_and_trainer_gpus(monkeypatch):
    monkeypatch.setattr(
        "opd.coordinator.base.CoordinatorBase._init_gpu_trace_sampler",
        lambda self: None,
    )
    config = OPDConfig(
        model=ModelConfig(path="student"),
        data=DataConfig(train_files="train.parquet", solution_key="solution"),
        rollout=RolloutConfig(gpu_ids="0"),
        trainer=TrainerConfig(gpu_ids="1"),
        algorithm=AlgorithmConfig(mode="opsd"),
        pipeline=PipelineConfig(n_step_off=NStepOffConfig(step_off=1)),
    )
    config.validate()

    coordinator = create_coordinator(config)

    assert isinstance(coordinator, StepOffCoordinator)
    assert coordinator.step_off == 1


@pytest.mark.parametrize("filename, model_path, batch_size", [
    ("opsd_gsm8k_0.5b_2gpu.yaml", "Qwen/Qwen2.5-0.5B-Instruct", 8),
    ("opsd_gsm8k_qwen3_4b_2gpu.yaml", "Qwen/Qwen3-4B", 4),
])
def test_two_gpu_opsd_example_selects_policy_gradient_and_native_lora(
    filename, model_path, batch_size,
):
    config_path = Path(__file__).resolve().parents[1] / "configs/examples" / filename
    config = OPDConfig.from_yaml(config_path)

    assert config.model.path == model_path
    assert config.algorithm.mode == "opsd"
    assert config.algorithm.opd.kl_loss_mode == "policy_gradient_kl"
    assert config.algorithm.opd.use_importance_sampling is True
    assert config.trainer.lora.native_lora is True
    assert config.rollout.gpu_ids == "0"
    assert config.trainer.gpu_ids == "1"
    assert config.teacher is None
    assert config.weight_sync.backend == "nccl"
    assert config.pipeline.n_step_off.step_off == 1
    assert config.trainer.mini_batch_size == config.trainer.batch_size
    assert config.trainer.batch_size == batch_size
    if model_path == "Qwen/Qwen3-4B":
        assert config.data.enable_thinking is False
        assert config.data.teacher_enable_thinking is True

    class CaptureTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return messages[0]["content"]

    rendered = format_prompt("What is 6 * 7?", CaptureTokenizer(),
                             config.data.prompt_template, config.data.enable_thinking)
    assert rendered == (
        "Problem: What is 6 * 7?\n\n"
        "Please reason step by step, and put your final answer within \\boxed{}."
    )

    baseline = OPDConfig.from_yaml(config_path, overrides=["pipeline.n_step_off.step_off=0"])
    assert baseline.pipeline.n_step_off.step_off == 0


def test_opsd_scores_and_records_teacher_logprobs_before_returning_rollout(monkeypatch):
    events = []
    result_queue = Queue()
    generated = {
        "input_ids": torch.tensor([[11, 12, 13, 21, 22]]),
        "attention_mask": torch.ones(1, 5, dtype=torch.bool),
        "prompt_lengths": torch.tensor([3]),
        "full_token_lists": [[11, 12, 13, 21, 22]],
        "student_logprobs": torch.tensor([[-0.4, -0.5]]),
    }

    class RolloutProxy:
        n_workers = 1
        _result_queues = [result_queue]

        def submit_generate(self, batch):
            assert batch["max_response_length"] == 2
            events.append("generate")
            result_queue.put(generated)

        def collect_generate(self):
            return result_queue.get()

        def submit_command(self, command, request):
            events.append(command)
            assert request["prompt_token_ids"] == [[31, 32, 21, 22]]
            result_queue.put({
                "_cmd": "score",
                "teacher_topk_logprobs": [torch.tensor([[-1.0], [-0.2], [-0.3]])],
                "teacher_topk_indices": [torch.tensor([[31], [21], [22]])],
                "teacher_token_logps": [torch.tensor([-1.0, -0.2, -0.3])],
            })

    config = OPDConfig(
        model=ModelConfig(path="student"),
        data=DataConfig(train_files="train.parquet", solution_key="solution"),
        rollout=RolloutConfig(train_max_tokens=2),
        algorithm=AlgorithmConfig(mode="opsd"),
    )
    mode = OPSDMode(
        rollout_proxy=RolloutProxy(), teacher_client=None, trainer_proxy=None,
        tracer=None, opd_config=config,
    )
    monkeypatch.setattr(mode, "_build_teacher_prompts", lambda gen, sols, problems:
                        ([[31, 32, 21, 22]], [2]))

    mode.async_generate({"solutions": ["answer"], "problem_texts": ["problem"]})
    scored_rollout = mode.wait_generate()

    assert events == ["generate", "score"]
    assert torch.equal(scored_rollout["student_logprobs"], generated["student_logprobs"])
    teacher = mode.resolve_teacher(mode.async_teacher(scored_rollout), {})
    assert events == ["generate", "score"]
    assert teacher["teacher_valid_mask"].tolist() == [[False, False, False, True, True]]
    assert teacher["teacher_token_logps"][0, 2:4].tolist() == torch.tensor([-0.2, -0.3]).tolist()


def test_opsd_capped_queued_rollouts_keep_their_own_self_scores():
    result_queue = Queue()
    score_prompts = []

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return messages[1]["content"]

        def encode(self, text, add_special_tokens=False):
            return [31 if "first solution" in text else 32]

    class RolloutProxy:
        n_workers = 1
        _result_queues = [result_queue]

        def submit_generate(self, batch):
            assert batch["max_response_length"] == 2
            first = int(batch["input_ids"][0, 0]) == 11
            response = [21, 22] if first else [23]
            result_queue.put({
                "input_ids": torch.tensor([[11, 12, 21, 22] if first
                                            else [13, 14, 23, 0]]),
                "attention_mask": torch.tensor([[1, 1, 1, 1] if first
                                                 else [1, 1, 1, 0]], dtype=torch.bool),
                "responses": torch.tensor([[21, 22] if first else [23, 0]]),
                "response_lengths": torch.tensor([len(response)]),
                "prompt_lengths": torch.tensor([2]),
                "full_token_lists": [([11, 12] if first else [13, 14]) + response],
                "student_logprobs": torch.tensor([[-0.4, -0.5] if first else [-0.6, 0.0]]),
            })

        def collect_generate(self):
            return result_queue.get()

        def submit_command(self, command, request):
            assert command == "score"
            prompt = request["prompt_token_ids"][0]
            score_prompts.append(prompt)
            n = len(prompt) - 1
            result_queue.put({
                "_cmd": "score",
                "teacher_topk_logprobs": [torch.full((n, 1), -0.2)],
                "teacher_topk_indices": [torch.tensor(prompt[1:], dtype=torch.int32).unsqueeze(-1)],
                "teacher_token_logps": [torch.full((n,), -0.2)],
            })

    config = OPDConfig(
        model=ModelConfig(path="student"),
        data=DataConfig(train_files="train.parquet", solution_key="solution"),
        rollout=RolloutConfig(train_max_tokens=2),
        algorithm=AlgorithmConfig(mode="opsd"),
    )
    mode = OPSDMode(rollout_proxy=RolloutProxy(), teacher_client=None,
                    trainer_proxy=None, tracer=None, opd_config=config,
                    tokenizer=Tokenizer())
    mode.async_generate({"input_ids": torch.tensor([[11, 12]]),
                         "solutions": ["first solution"], "problem_texts": ["first"]})
    mode.async_generate({"input_ids": torch.tensor([[13, 14]]),
                         "solutions": ["second solution"], "problem_texts": ["second"]})

    first = mode.wait_generate()
    first_teacher = mode.resolve_teacher(mode.async_teacher(first), {})
    second = mode.wait_generate()
    second_teacher = mode.resolve_teacher(mode.async_teacher(second), {})

    assert score_prompts == [[31, 21, 22], [32, 23]]
    assert first_teacher["teacher_valid_mask"].tolist() == [[False, False, True, True]]
    assert second_teacher["teacher_valid_mask"].tolist() == [[False, False, True, False]]
    torch.testing.assert_close(first["student_logprobs"], torch.tensor([[-0.4, -0.5]]))
    torch.testing.assert_close(second["student_logprobs"], torch.tensor([[-0.6, 0.0]]))


@pytest.mark.parametrize("eval_flags", [{"eval": True}, {"eval_n_samples": 2}])
def test_opsd_validation_rollout_skips_teacher_scoring(eval_flags):
    generated = {"responses": torch.tensor([[1, 2]])}

    class RolloutProxy:
        def submit_generate(self, batch):
            pass

        def collect_generate(self):
            return generated

        def submit_command(self, command, request):
            pytest.fail("validation rollout must not request teacher scoring")

    mode = OPSDMode(
        rollout_proxy=RolloutProxy(), teacher_client=None, trainer_proxy=None,
        tracer=None,
    )
    mode.async_generate(dict(eval_flags))
    assert mode.wait_generate() is generated


def test_policy_gradient_trainer_uses_live_student_forward_and_updates_weight():
    class Student(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.logprob = torch.nn.Parameter(torch.tensor(-0.7))

        def forward(self, input_ids, **kwargs):
            return self.logprob.expand(input_ids.size(0), input_ids.size(1) - 1, 1)

    student = Student()
    optimizer = torch.optim.SGD(student.parameters(), lr=0.1)
    trainer = OPDTrainer.__new__(OPDTrainer)
    trainer.kl_config = KLConfig(mode="policy_gradient_kl", use_importance_sampling=True)
    trainer._use_decoupled_ppo = False
    trainer._backend = type("Backend", (), {"kl_chunk_size": 32})()
    batch = {
        "input_ids": torch.tensor([[11, 12, 21, 22]]),
        "attention_mask": torch.ones(1, 4, dtype=torch.bool),
        "response_mask": torch.tensor([[False, False, True, True]]),
        "prompt_lengths": torch.tensor([2]),
        "teacher_token_logps": torch.tensor([[-1e10, -0.2, -0.3, -1e10]]),
        "student_logprobs": torch.tensor([[-0.7, -0.7]]),
    }
    micro_batch = dict(batch, max_prompt=2, actual_max_len=4, seq_len=4)

    loss, n_tokens, _ = trainer.forward_and_loss_fn(student, micro_batch, torch.device("cpu"))
    assert n_tokens == 2
    assert loss.requires_grad
    before = student.logprob.item()
    loss.backward()
    optimizer.step()
    assert student.logprob.item() > before


def test_policy_gradient_training_rejects_missing_rollout_logprobs():
    trainer = OPDTrainer.__new__(OPDTrainer)
    trainer.kl_config = KLConfig(mode="policy_gradient_kl", use_importance_sampling=True)
    trainer._use_decoupled_ppo = False

    class Backend:
        def _prepare_train_batch(self, batch):
            return {
                "input_ids": torch.tensor([[11, 12, 21]]),
                "attention_mask": torch.ones(1, 3, dtype=torch.bool),
                "response_mask": torch.tensor([[False, False, True]]),
                "prompt_lengths": torch.tensor([2]),
                "max_prompt": 2,
                "actual_max_len": 3,
                "batch": {"teacher_token_logps": torch.tensor([[-1e10, -0.2, -1e10]])},
                "teacher_topk_logps": None,
                "teacher_topk_indices": None,
            }

        def _run_train_step(self, *args, **kwargs):
            pytest.fail("trainer should reject missing rollout logprobs before forward")

    with pytest.raises(ValueError, match="student_logprobs"):
        trainer.train_step({}, Backend())
