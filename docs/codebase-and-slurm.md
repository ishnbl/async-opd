# Codebase and Slurm guide for research changes

This guide maps the current implementation and gives a single-node Slurm path
for running it. It focuses on OPD because the rollout distribution, teacher
scoring, importance weights, and loss are the parts most relevant to studying
importance sampling, baselines, variance, and rollout quantization.

The code and comments include a Thinking Machines-style policy-gradient KL
path. I interpret “TIM” in the research goal as that line of work; if TIM means
a different paper or method, the implementation map below is still useful, but
the method-specific comparison should be adjusted.

## How the repository is organized

| Area | Files | What it owns |
| --- | --- | --- |
| CLI and config | `opd/cli/train.py`, `opd/utils/config.py` | Parse the run command, validate YAML, apply `--set` overrides, create output files. |
| Pipeline and coordinator | `opd/pipeline.py`, `opd/coordinator/` | Start child processes, schedule batches, route results, manage staleness/evaluation/checkpoints. |
| Rollout | `opd/rollout/`, especially `opd/rollout/vllm/` | Load the student in vLLM, generate tokens, capture sampled-token or support logprobs, receive new student weights. |
| Teacher/reference | `opd/worker/teacher/` | Score generated sequences and return teacher logprobs/top-k support. |
| Trainer | `opd/trainer/opd.py`, `opd/trainer/base_trainer.py`, `opd/trainer/fsdp/`, `opd/trainer/megatron/` | Recompute current student logprobs, calculate the objective, backpropagate, update parameters. |
| Objectives | `opd/loss/kl.py`, `opd/loss/ppo.py`, `opd/loss/advantages.py`, `opd/loss/grpo.py` | KL variants, PG-KL/PPO surrogate, advantage utilities, GRPO loss. |
| Batch/data alignment | `opd/data/` | Format prompts and align response masks, teacher scores, rollout scores, and multi-sample tensors. |
| Weight exchange | `opd/worker/weight_sync.py`, `opd/rollout/vllm/` | Transfer trainer parameters to rollout workers, normally over NCCL. |

Useful existing documentation: [architecture](architecture.md),
[training modes](training-modes.md), [configuration](configuration.md), and
[loss/logit chunking](loss-chunking.md).

## OPD run, end to end

The default path is a coordinator process plus independent role workers.

1. `opd/cli/train.py` loads `OPDConfig`, creates `results/<config path>/`, and
   calls `create_coordinator` in `opd/pipeline.py`.
2. The coordinator starts the teacher, rollout, and trainer workers. Local mode
   uses Python multiprocessing with the `spawn` start method. The coordinator
   is mostly CPU-side orchestration; the model workers own CUDA contexts.
3. A data iterator yields prompt batches. In OPD, rollout workers run the
   student through vLLM and return response token IDs, lengths, masks, and any
   logprobs/support requested by the selected objective.
4. The coordinator sends those sequences to the teacher scoring service. The
   teacher returns teacher token logprobs and/or top-k logprobs for the
   generated positions. The coordinator pads/adapts these artifacts and forms
   the trainer batch.
5. The trainer computes fresh student logits/logprobs for the batch. The
   FSDP/Megatron backend performs forward/backward, optimizer steps, and
   checkpointing. OPD's single loss dispatch point is
   `OPDTrainer._compute_loss()` in `opd/trainer/opd.py`; it calls
   `compute_kl_loss()` in `opd/loss/kl.py`.
6. Updated trainer weights are sent to the vLLM student. The rollout then
   generates under the updated policy, subject to the selected scheduler's
   staleness rules.

The ordinary step-off scheduler overlaps rollout and training up to the
configured `step_off` distance. `step_off: 0` is the easiest setting for
checking mathematical and tensor alignment. Fully async and streaming paths
have different queueing/weight-version behavior; start with synchronous or
small step-off for a new loss.

### What data is available to a loss?

The loss path can receive different tensors depending on `algorithm.opd.kl_loss_mode`:

| Signal | Source | Common use |
| --- | --- | --- |
| Current student logprob, `log πθ(y_t)` | Trainer forward on the training batch | Differentiable policy term. |
| Behavior/rollout logprob, `log πold(y_t)` | Requested from vLLM during generation | PPO ratio and off-policy correction when rollout and training policies differ. |
| Teacher logprob, `log πT(y_t)` | Teacher scoring worker | Token-level KL or PG-KL advantage. |
| Teacher top-k logprobs and token IDs | Teacher scoring worker | Sparse forward/reverse/skewed KL objectives. |
| Rollout student top-k support | vLLM rollout | Top-k objectives such as THUNLP default and rollout-support reverse KL. |
| Multi-sample token IDs and teacher/old logprobs | Rollout plus teacher over the candidate samples | Multi-sample PG-KL/forward-KL and MOF variants. |
| Response mask | Generation/batch adaptation | Excludes prompt, padding, and non-response positions from loss/reductions. |

For the current `policy_gradient_kl`, the code constructs
`advantage_t = log πT(y_t) - log πold(y_t)` by default and the PPO ratio
`πθ(y_t) / πold(y_t)`. With `pg_online_advantage: true`, it substitutes the
detached current student logprob as the advantage baseline. The shared PPO
implementation is in `opd/loss/ppo.py`; it handles clipping, optional
decoupled ratios/behavior weights, M2PO bounds, masks, and reduction.

One important constraint for estimator experiments: the data requested from
rollout is conditional on the configured loss. For example,
`opd/coordinator/process_lifecycle.py` enables sampled student logprobs for
`policy_gradient_kl` when importance sampling is on. A new estimator that needs
additional behavior-policy statistics must thread those through rollout,
coordinator/batch adaptation, trainer launch config, and loss dispatch. Do not
assume every mode already carries every logprob.

## GPU placement

In a typical eight-GPU single-node OPD config, each GPU role is:

| Local GPU IDs | Role | Model/work |
| --- | --- | --- |
| `0` | Teacher | Teacher vLLM scoring model. It may return only requested token/top-k scores. |
| `1,2,3` | Rollout | Student vLLM generation replicas/workers. These hold KV cache and generation state. |
| `4,5,6,7` | Trainer | Four FSDP ranks, each rank assigned one GPU; they jointly train/shard the student. |
| CPU | Coordinator | Scheduling, data iteration, queues, logging, and process lifecycle. |

This is the allocation in `configs/examples/opd_qwen3_1.7b.yaml`. Config GPU
IDs are passed into role processes, which set `CUDA_VISIBLE_DEVICES`; FSDP
trainer ranks each expose one configured GPU and use it as local `cuda:0`.
Therefore configs must agree with the device numbering visible to those child
processes.

On Slurm, the simplest supported arrangement is to request all GPUs on one
node and keep config IDs `0..N-1`. Slurm GPU visibility and device numbering
vary by site. Before a run, inspect `CUDA_VISIBLE_DEVICES` and `nvidia-smi` in
the allocation. If your allocation exposes a remapped/subset device list, make
the YAML `gpu_ids` match the numbering this code can address, or adapt the
worker GPU mapping; blindly copying physical host IDs into `gpu_ids` may target
the wrong device or fail inside a restricted allocation. Start with one node:
the ordinary local multiprocessing setup is not a multi-node Slurm launcher.

Memory is split by role, so a teacher OOM, rollout KV-cache OOM, and trainer
activation/optimizer OOM need different fixes. Teacher sizing is controlled by
`teacher.vllm.*`, rollout by `rollout.vllm.*` plus generation context/concurrency,
and trainer by `trainer.micro_batch_size`, sequence packing, dtype, and
`trainer.kl_chunk_size`. Rollout quantization is a vLLM rollout setting and
does not quantize the FSDP training weights.

## Single-node Slurm setup

The exact GPU request syntax, partition, account, and environment activation
are cluster-specific. This template requests eight GPUs on one node; adapt the
SBATCH resource lines to your site. The Qwen3 example expects those eight GPUs
split 1 teacher + 3 rollout + 4 trainer.

```bash
#!/bin/bash
#SBATCH --job-name=async-opd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

cd /path/to/async-opd
source /path/to/conda/etc/profile.d/conda.sh
conda activate opd

# Keep model/dataset caches on a filesystem with enough capacity.
export HF_HOME=/path/to/shared-or-scratch/huggingface
export HF_HUB_CACHE="$HF_HOME/hub"
export TOKENIZERS_PARALLELISM=false

echo "host=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi

python -m opd.cli.train \
  --config configs/examples/opd_qwen3_1.7b.yaml \
  --overwrite
```

Submit and inspect it with:

```bash
sbatch run_opd.sbatch
squeue -u "$USER"
```

The standard 8-GPU example points to a Hugging Face dataset and model IDs, so
the compute node needs network access or a populated cache. Some configs refer
to local parquet files that are not shipped in the repository; generate or
stage those files on shared storage before submission. Keep checkpoints and
results on storage sized for the run. The CLI writes logs and metrics under
`results/examples/opd_qwen3_1.7b/` for this config.

For a one-step setup smoke, replace the Python command with:

```bash
python -m opd.cli.train \
  --config configs/examples/opd_qwen3_1.7b.yaml \
  --overwrite \
  --set trainer.total_steps=1 trainer.total_epochs=1 eval.freq=-1 \
        eval.before_train=false trainer.save_freq=-1 \
        pipeline.n_step_off.step_off=0
```

The CLI checks for a dirty git tree for ordinary experiment configs. Commit
code/config changes before launching a research run, or use `--allow-dirty`
when deliberately measuring an uncommitted change. The CLI also frees stale
processes on configured GPU IDs before starting, so only point the config at
GPUs assigned to this job.

### Smaller 4-GPU allocation

For an initial run on four GPUs, use
`configs/examples/opd_gsm8k_0.5b_4gpu.yaml`. Inspect that file's role split and
data settings before submitting; it places a small teacher and student rollout
alongside the trainer with a different model/data pairing. The same one-step
override pattern applies. Do not use the eight-GPU config while requesting
four GPUs.

## How to add an objective, estimator, or baseline

For an experiment that only changes existing knobs, first copy a config and
select an existing mode. Relevant settings include `kl_loss_mode`,
`use_importance_sampling`, `pg_online_advantage`, `use_decoupled_loss`,
`behave_imp_weight_cap`, `pg_clip_eps`, and `pg_m2po_budget` under
`algorithm.opd`. Keep a copy of the exact config and git commit with every run.

For a new method, trace and update the full path:

1. **Define the estimator precisely.** Write down target and behavior policies,
   token/sample axes, detached terms, baseline, clipping/truncation, and final
   reduction. State what distribution each logged quantity estimates.
2. **Add required rollout artifacts.** Update vLLM request flags/extraction in
   `opd/rollout/vllm/` if the method needs behavior logprobs, top-k support,
   samples, or quantization metadata. Current logprob request selection starts
   in `ProcessLifecycle._apply_rollout_logprob_flags()`.
3. **Preserve alignment.** Carry artifacts through the coordinator's OPD
   submission/scoring/adaptation code in `opd/coordinator/opd_mode.py` and
   `opd/data/batch_utils.py`. Check shifting, left padding, response masks,
   EOS, sequence packing, and multi-sample dimensions.
4. **Implement the math.** Put shared math in `opd/loss/kl.py` or
   `opd/loss/ppo.py` when it reuses the PPO surrogate. Keep the trainer call
   centered on `OPDTrainer._compute_loss()` where possible. Advantage helpers
   live in `opd/loss/advantages.py`.
5. **Thread config through process boundaries.** Add fields and validation in
   `opd/utils/config.py`, then carry them through the typed payload/builders in
   `opd/launch_specs.py` and `opd/trainer/config.py`, to `KLConfig` and the
   loss dispatch. Child workers receive serialized launch specs, so a field
   added only to YAML will not automatically reach the trainer.
6. **Log estimator diagnostics.** The OPD trainer already collects ratios,
   log-ratios, advantages, clipping fractions, and decoupled behavior weights
   from `loss.pg_stats`. Add diagnostics for weight tails/ESS, baseline scale,
   variance, and effective sample count where those are meaningful for the
   proposed method.
7. **Start synchronous and inspect batches.** Use `step_off: 0`, short
   sequences, a small dataset, and one or a few steps. Then compare eager
   reference math against the production/chunked and packed paths before
   turning on overlap.

Existing relevant variants include `policy_gradient_kl`,
`multi_sample_policy_gradient_kl`, `multi_sample_forward_kl`,
`reverse_kl_rollout_student_topk`, `thunlp_opd_default_loss`, and `mof_opd`.
The PG-KL implementation docstring calls out the Thinking Machines-style
advantage/ratio form. That gives you a runnable baseline for a study, but
verify the equations, sampling distribution, and reduction against the exact
reference you intend to reproduce.

## Quantized rollouts and policy mismatch

The rollout config has `rollout.quantization`; the vLLM worker currently has
special handling for `fp8` and `fp8_blockwise` in
`opd/rollout/vllm/utils.py` and `opd/rollout/vllm/fp8.py`. These code paths
patch vLLM's weight-loading/update behavior because trainer weights are
synchronized into the rollout engine. FP8 paths require supported GPU
hardware; the code checks for compute capability 8.9 or newer. Other vLLM
quantization modes depend on the pinned vLLM version, model format, and the
ability to receive repeated weight updates. Confirm initialization and at
least one post-update sync before a long experiment.

Quantized generation may make the rollout behavior policy differ from the
full-precision trainer policy even when weights are nominally the same. For
importance-sampling research, treat that as a first-class experimental factor:
record rollout quantization and sampling settings, ensure stored old logprobs
are computed by the actual behavior engine, monitor log-ratio/weight tails and
effective sample size, and compare against an unquantized rollout. A measured
policy mismatch can reflect quantized inference as well as scheduler
staleness.

## Outputs and useful inspection points

For each run, inspect:

- `run.log`: process startup, GPU placement, vLLM initialization, worker sync,
  and errors.
- `log.jsonl`: per-step metrics and configuration/run records.
- `trace.json`: pipeline stage overlap and wait time; open it with a Perfetto
  trace viewer.
- `checkpoints/`: saved trainer checkpoints, when saving is enabled.
- `validation_outputs/`: generated validation samples when configured.

For implementation navigation, start at
`opd/coordinator/process_lifecycle.py` for worker topology,
`opd/coordinator/step_off.py` for the bounded-overlap scheduler,
`opd/coordinator/opd_mode.py` for the OPD batch/scoring route,
`opd/trainer/opd.py` for training loss input assembly, and
`opd/loss/kl.py` / `opd/loss/ppo.py` for estimator math.
