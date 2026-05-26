# Installation

## Requirements

* Python ≥ 3.10, CUDA 12.1+
* `ffmpeg` ≥ 4.4 on the system path
* One GPU with ≥ 24 GB memory

## 1. Clone and install dependencies

```bash
git clone <repo-url> omniretriever
cd omniretriever
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
export PYTHONPATH=$PWD/src:$PYTHONPATH
```

## 2. Download the WAVE-7B backbone

OmniRetriever-7B is a LoRA on top of the public WAVE-7B backbone. Point
`WAVE_HOME` at the directory containing `WAVE-7B/` and the BEATs
checkpoint:

```bash
export WAVE_HOME=/path/to/wave-7b-root
# Expected layout:
#   $WAVE_HOME/
#   ├── WAVE-7B/                                          ← HF-format weights
#   │   ├── config.json
#   │   └── model-00001-of-0000N.safetensors
#   └── BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt   ← audio encoder
```

## 3. Pull the adapter and benchmark from the Hub

```bash
huggingface-cli download YunzeLiu/OmniRetriever-7B    --local-dir adapters/omniretriever-7b
huggingface-cli download YunzeLiu/OmniRetriever-Bench --repo-type dataset \
    --local-dir benchmark
```

## 4. Verify

```bash
python -c "from omniretriever import OmniRetriever, recall_at_k; print('OK')"
```
