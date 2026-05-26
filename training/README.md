# OmniRetriever training framework

Reproducible training code for OmniRetriever-7B, the unified AVT retriever
described in *OmniRetriever: Any-to-Any Audio-Video-Text Retrieval via
Fusion-as-Teacher Distillation*.

> This is for researchers who want to **re-train** or **ablate** the model.
> For inference-only use of the released LoRA adapter, see the top-level
> [`README.md`](../README.md) and the model card on Hugging Face.

## Layout

```
training/
├── README.md                ← this file
├── train.sh                 ← canonical launcher (paper recipe)
├── requirements.txt         ← deepspeed, peft, accelerate, ...
├── configs/
│   └── ds_zero0.json        ← DeepSpeed ZeRO-0 config used in the paper
├── qwenvl/                  ← training framework
│   ├── train/               ← Trainer, sampler, argument parser, entry point
│   ├── data/                ← data pipeline (LazySupervisedDataset + collator)
│   └── model/qwen2_5_omni/  ← Qwen2.5-Omni / WAVE-7B fork with our losses
└── third-party-licenses/    ← upstream attributions (REQUIRED)
```

## Method recap

Three losses on top of the WAVE-7B backbone (see paper §3):

| Loss | Where it lives | How to toggle |
| --- | --- | --- |
| `L_A` — pairwise InfoNCE (T-V, T-A, V-A) | `qwenvl/model/.../modeling_qwen2_5_omni.py` | always on with `--train_classify True` |
| `L_D` — **fusion-as-teacher distillation** *(main contribution)* | same file (`anchor_embed.detach()` in the joint forward) | `--train_classify True --classify_type all_layer` |
| `L_T` — Tuple-InfoNCE refinement | same file | `--use_tuple_infonce True` |

Final objective: `L_A + L_D + L_T` with uniform weights. The default
`train.sh` runs this recipe.

## 0. Prerequisites

* Linux, Python ≥ 3.10, CUDA 12.1+
* **4 × H100/A100 80 GB** (paper hardware) — single-GPU works for small ablations
* `ffmpeg` ≥ 4.4 on the system path
* ~500 GB free disk for the training corpus and checkpoints

## 1. Install dependencies

```bash
# from the repo root:
pip install -r requirements.txt           # inference deps
pip install -r training/requirements.txt  # adds deepspeed + peft helpers
export PYTHONPATH=$PWD/training:$PYTHONPATH
```

## 2. Prepare the backbone

Download the public WAVE-7B backbone (HuggingFace layout) and the BEATs
audio-encoder checkpoint, then expose them via env vars:

```bash
export WAVE_PATH=/path/to/WAVE-7B
export BEATS_PATH=/path/to/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt
```

## 3. Prepare the training manifest

`DATA_PATH` is a JSONL file, one record per line:

```json
{
  "id": "...",
  "conversations": [
    {"from": "human", "value": "<image>\nPlease describe the video."},
    {"from": "gpt",   "value": "<the dense caption that is the training target>"}
  ],
  "video": "/abs/path/to/clip.mp4",
  "audio": "/abs/path/to/clip.wav"
}
```

Notes:
- `<image>` is the placeholder for the visual+audio stream; the loader
  re-routes it to the multimodal encoder.
- The video file can be MP4 with muxed audio, in which case `audio` may
  point at the same MP4.
- The full curation pipeline (4-stage filter, deduplication) is described
  in paper §E.

## 4. Train

```bash
WAVE_PATH=/path/to/WAVE-7B \
BEATS_PATH=/path/to/BEATs.pt \
DATA_PATH=/path/to/omniretriever_1m.jsonl \
bash training/train.sh
```

This launches DeepSpeed ZeRO-0 on 4 GPUs, batch 8 × 8 grad-accum (effective
256), LR 1e-5 cosine, 1 epoch — the exact recipe used for the released
LoRA. Tweakable env vars are listed at the top of `train.sh`.

Expected wall-clock on **4 × H100**: ~109 GPU-hours total (~27 h elapsed)
for the full ~1 M-triple corpus.

### Ablations

| Run | Override |
| --- | --- |
| Pairwise-only baseline (paper Table 4 row 1) | `USE_TUPLE_INFONCE=False bash train.sh` + further set `--classify_type single` |
| `L_A + L_D` (no Tuple-InfoNCE) | `USE_TUPLE_INFONCE=False bash train.sh` |
| LoRA rank sweep | `LORA_R=8 LORA_ALPHA=16 ...` |
| Single-GPU smoke test | `NUM_GPUS=1 BATCH_SIZE=2 GRAD_ACCUM=2 bash train.sh` |

### Output

```
${OUTPUT_DIR}/
├── adapter_model.safetensors   ← LoRA weights + classify_linear + beats_ln/proj
├── adapter_config.json
└── tokenizer.*                  ← carried over from the backbone
```

Plug this directly into the inference pipeline:

```python
from omniretriever import OmniRetriever
model = OmniRetriever.from_pretrained(
    base_model=os.environ["WAVE_PATH"],
    adapter="output/omniretriever_7b",     # this run's checkpoint
)
```

## Attribution

The training stack adapts code from:

| Upstream | Adopted parts | License file |
| --- | --- | --- |
| [Qwen2.5-Omni](https://github.com/QwenLM/Qwen2.5-Omni) | model definition & multimodal processor | `third-party-licenses/qwen25omni.txt` |
| [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL) | training scaffold (`train_qwen.py`, arguments, sampler) | `third-party-licenses/qwenvl25.txt` |
| [HuggingFace Transformers](https://github.com/huggingface/transformers) | base modeling utilities | `third-party-licenses/transformers.txt` |
| [video-SALMONN-2](https://github.com/bytedance/video-SALMONN-2) | initial trainer hooks | `third-party-licenses/video-salmonn-2.txt` |

All upstream files retain their original Apache-2.0 license; our additions
are released under the same license (with the no-biometric-deployment
clause described in the repository-level README).

## Limitations

- **Backbone availability.** You must obtain WAVE-7B and the BEATs
  checkpoint separately; neither is redistributed here.
- **Training data.** The exact ~1 M-triple `OmniRetriever-1M` corpus we
  used is not released; you can train on any AVT corpus that matches the
  manifest schema above.
- **Single-machine only.** The launcher assumes one node; multi-node DS
  configuration is straightforward but not provided.
