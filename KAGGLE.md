# Running SETU-RAG on Kaggle

Kaggle Notebooks give you a **free GPU** (2× T4 16 GB, or 1× P100) — actually *more* VRAM than a
single Colab T4. But a few Kaggle-specific settings differ from Colab. Follow these in order.

---

## 0. What "running" means here
This repo is a **dissertation scaffold**: the novel logic (CMI, RRF, multi-view, LID fusion,
CMI-conditioned prompts/TTS descriptions, WER/CER) is **implemented and runs immediately**; the
model-loading wrappers are marked `# TODO`. So on Kaggle you can:
- **Now:** run the implemented logic + notebooks (no model downloads needed).
- **Full pipeline:** fill the `# TODO` stubs (each documents its exact model call), then run end-to-end.

---

## 1. Create the notebook + turn on GPU and Internet
1. kaggle.com → **Create → New Notebook**.
2. Right sidebar **Settings**:
   - **Accelerator → GPU T4 ×2** (or P100).
   - **Internet → On**  ← *required* to pip-install and download Hugging Face models (off by default).
3. (Language: Python.)

> Quota: ~30 GPU-hours/week. `/kaggle/working` persists across "Save Version" runs (≤ ~20 GB);
> `/kaggle/input` is read-only.

---

## 2. Get the code into the notebook (pick one)

**Option A — upload the zip as a Dataset (simplest, no GitHub needed)**
1. **Create → New Dataset →** upload `setu-rag.zip` → name it e.g. `setu-rag`.
2. In the notebook sidebar: **Add Input →** your `setu-rag` dataset. It mounts read-only at
   `/kaggle/input/setu-rag/`.
3. First cell:
   ```python
   import shutil, zipfile, os
   src = "/kaggle/input/setu-rag/setu-rag.zip"          # adjust if your dataset nests it
   if os.path.exists(src):
       zipfile.ZipFile(src).extractall("/kaggle/working/")
   else:                                                 # dataset uploaded already-unzipped
       shutil.copytree("/kaggle/input/setu-rag", "/kaggle/working/setu-rag", dirs_exist_ok=True)
   %cd /kaggle/working/setu-rag
   ```

**Option B — git clone (if you push the repo to GitHub)**
```python
!git clone https://github.com/<you>/setu-rag.git /kaggle/working/setu-rag
%cd /kaggle/working/setu-rag
```

---

## 3. Add your Hugging Face token (for model downloads)
Some models are gated or rate-limited (Sarvam-1, Aya-Expanse, IndicConformer, Parler-TTS).
1. **Add-ons → Secrets →** add a secret named `HF_TOKEN` (value = your token from huggingface.co/settings/tokens).
2. In a cell:
   ```python
   from kaggle_secrets import UserSecretsClient
   import os
   os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
   from huggingface_hub import login; login(os.environ["HF_TOKEN"])
   # cache downloads to working dir so they survive "Save Version"
   os.environ["HF_HOME"] = "/kaggle/working/hf_cache"
   ```
   Then request access on each gated model's HF page once (instant for most).

---

## 4. Install dependencies (Internet must be On)
```python
!pip -q install -r requirements.txt
# from-source bits (see requirements.txt notes):
!pip -q install git+https://github.com/huggingface/parler-tts.git    # TTS
# IndicConformer ASR (NeMo) + IndicTrans2 toolkit + IndicLID: install per their GitHub READMEs
```
If a heavy install (NeMo) is slow, install only what a given experiment needs.

---

## 5. Run it
**Quick smoke test (no downloads):**
```python
!python - <<'PY'
import sys; sys.path.insert(0,".")
from setu_rag.front_end.cmi import compute_cmi
from setu_rag.front_end.language_id import LanguageIdentifier
from setu_rag.router.adaptive_router import decide_route
lid = LanguageIdentifier().load()
t = lid.tag("mera refund kab tak aayega order cancel kiya")
c = compute_cmi(t); fr = sum(x.script=='Latn' for x in t)/len(t)
print("CMI", c.cmi, "matrix", c.matrix_lang, "->", decide_route(c, fr).route.value)
PY
```
**Notebooks:** open `notebooks/setu_rag_colab.ipynb` (text) and `notebooks/setu_rag_speech_colab.ipynb`
(speech) — they run the same on Kaggle. Or use this file's companion `notebooks/setu_rag_kaggle.ipynb`.

**Build the index / demo** (after wiring the `# TODO` model stubs):
```python
!python scripts/build_index.py --faqs data/faqs.sample.jsonl --out /kaggle/working/index
```

---

## 6. Gradio demo on Kaggle
`scripts/serve.py` (text) and `scripts/serve_voice.py` (speech) call `demo.launch()`. On Kaggle add
`share=True` to get a public URL you can open in your browser — and the **microphone works through that
shared link** (Kaggle's own output pane can't access your mic):
```python
# in serve.py / serve_voice.py
demo.launch(share=True)
```

---

## 7. Two-GPU tip (2× T4 = 32 GB)
You have more room than Colab. Options:
- Keep it simple: use `cuda:0` for everything (the default; on-demand loading already fits 16 GB).
- Or split load: put the generator on `cuda:0` and the embedder/ASR on `cuda:1`
  (`SETTINGS.device` is the single knob; set per-component devices in `config.py`), or pass
  `device_map="auto"` to `from_pretrained` so 🤗 Accelerate shards big models across both T4s.

---

## Common gotchas
| Symptom | Fix |
|---|---|
| `pip`/downloads hang or fail | **Internet → On** in Settings (off by default). |
| Gated model 401/403 | Add `HF_TOKEN` secret + accept the model's license on its HF page. |
| Re-downloads every run | Set `HF_HOME=/kaggle/working/hf_cache` and **Save Version** to persist. |
| Out of memory | Keep `load_in_4bit=True`; rely on on-demand loading; FAISS stays on CPU. |
| Mic does nothing | Use `launch(share=True)` and open the public link in your own browser. |
| Session resets | GPU sessions are time-limited; **Save Version** to checkpoint `/kaggle/working`. |
