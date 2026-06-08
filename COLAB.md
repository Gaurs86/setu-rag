# Running SETU-RAG on Google Colab

Colab is the **primary target** for this project — everything is sized for a single **T4 (16 GB)** on the
free tier. Internet is on by default, so there's less setup than Kaggle.

The repo ships one notebook: **`notebooks/setu_rag.ipynb`** — speech-to-speech end-to-end
(setup → build pipeline → Gradio mic/upload UI → CS-RAGAS eval table → optional AI4Bharat upgrades).

This guide covers the environment around it (GPU, getting the code in, tokens, persistence).

---

## 0. TL;DR — fastest path
Set the GPU (step 1), get the code in (step 2), then open
**`notebooks/setu_rag.ipynb` → Runtime → Run all.** It clones the repo, installs deps, builds the
pipeline, and opens a Gradio mic/upload UI for speech-to-speech — **no HF token needed** for the
core stack (BGE-M3 + BGE-reranker + Qwen2.5-3B-Instruct are un-gated). Everything below is the
manual version.

The core text pipeline runs end-to-end out of the box. Each model also has a graceful fallback, so it
still runs offline/CPU (with stand-in models) for a quick structure check. The AI4Bharat front-end
(IndicLID/IndicXlit/IndicTrans2) and the speech layer are **optional add-ons**.

---

## 1. Turn on the GPU
**Runtime → Change runtime type → Hardware accelerator → T4 GPU → Save.** Verify:
```python
!nvidia-smi -L          # should show "Tesla T4"
```
> Free tier ≈ one T4, ~12 h max session, ~90 min idle timeout, GPU availability varies.
> Colab Pro gives more reliable/longer GPU and more RAM if you hit limits.

---

## 2. Get the code into the runtime (pick one)

**Option A — upload the zip (quickest)**
Click the **Files** panel (left) → upload `setu-rag.zip`, then:
```python
import zipfile; zipfile.ZipFile("setu-rag.zip").extractall(".")
%cd setu-rag
```
> Colab storage is **ephemeral** — it's wiped when the runtime disconnects. Fine for a session;
> use Option B if you want your edits to persist.

**Option B — Google Drive (persistent)**
```python
from google.colab import drive; drive.mount('/content/drive')
# put setu-rag/ (or the zip) in your Drive once, then:
%cd /content/drive/MyDrive/setu-rag
```

**Option C — git clone (if you push to GitHub)**
```python
!git clone https://github.com/<you>/setu-rag.git
%cd setu-rag
```

---

## 3. Hugging Face token (for model downloads)
Use **Colab Secrets** (the 🔑 key icon in the left sidebar) → add `HF_TOKEN` → enable for this notebook.
```python
from google.colab import userdata
import os
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
from huggingface_hub import login; login(os.environ["HF_TOKEN"])
# cache models to Drive so you don't re-download each session (optional):
# os.environ["HF_HOME"] = "/content/drive/MyDrive/hf_cache"
```
Needed for gated/rate-limited models (Sarvam-1, Aya-Expanse, IndicConformer, Parler-TTS); accept each
model's license on its HF page once.

---

## 4. Install dependencies
```python
!pip -q install -r colab_requirements.txt    # minimal, un-gated, runs the text pipeline
```
That's all you need for the core system. Optional extras (each is **real-with-fallback** — install
to upgrade a stage, skip to let it degrade gracefully):
```python
# !pip -q install ai4bharat-transliteration                       # IndicXlit native-script view
# !pip -q install git+https://github.com/AI4Bharat/IndicLID.git   # token LID (all 22 langs, romanized)
# !pip -q install IndicTransToolkit                               # IndicTrans2 English-pivot views
# !pip -q install soundfile librosa git+https://github.com/huggingface/parler-tts.git   # speech out
# IndicConformer (NeMo) ASR: per its GitHub/HF README (Whisper is the auto-fallback)
```
Then enable the translation-pivot views explicitly:
```python
rag = build_pipeline(enable_translation=True)   # English-pivot + matrix-canonical views
```

---

## 5. Run it
**Smoke test (no downloads):**
```python
import sys; sys.path.insert(0, ".")
from setu_rag.front_end.language_id import LanguageIdentifier
from setu_rag.front_end.cmi import compute_cmi
from setu_rag.router.adaptive_router import decide_route
lid = LanguageIdentifier().load()
for q in ["mera refund kab tak aayega order cancel kiya tha",
          "why is my coupon not applying to the cart"]:
    t = lid.tag(q); c = compute_cmi(t); fr = sum(x.script=='Latn' for x in t)/len(t)
    print(f"CMI={c.cmi:.2f} matrix={c.matrix_lang} -> {decide_route(c, fr).route.value}  | {q[:40]}")
```
**Full pipeline (real models, answers questions):**
```python
import sys, os; sys.path.insert(0, os.getcwd())
from setu_rag.app import build_pipeline
rag = build_pipeline()                       # downloads weights once, builds the index
tr = rag.answer("mera refund kab tak aayega, maine order cancel kiya tha")
print(tr.route, tr.cmi, tr.matrix_lang, "->", tr.answer)
print("retrieved:", tr.retrieved)
```
Or from a terminal cell: `!python scripts/build_index.py` (add `--offline` for the no-GPU check).
Your own KB: `build_pipeline(faq_path="/content/my_faqs.jsonl")` (one `{"question","answer","lang"}` per line).

**Evaluation table** (CS-RAGAS + WER/CER over the sample code-switched pairs):
```python
!python scripts/run_eval.py --offline    # deterministic, no downloads
!python scripts/run_eval.py              # real models if available
```

---

## 6. Gradio demo (with working mic)
`scripts/serve.py` (text) and `scripts/serve_voice.py` (speech) call `demo.launch()`. Add `share=True`
for a public link, or Colab renders the UI inline. The **microphone works** either way in your browser:
```python
# in serve.py / serve_voice.py:  demo.launch(share=True)
!python scripts/serve_voice.py
```

---

## 7. Staying inside 16 GB (single T4)
Already handled by the design — keep these defaults:
- `SETTINGS.load_in_4bit = True` (4-bit generator via bitsandbytes).
- On-demand loading: ASR → text core → TTS run sequentially per turn and are freed between stages.
- FAISS stays on **CPU**; IndicTrans2 uses the distilled 200M variants.

Watch VRAM with `!nvidia-smi`. If you ever OOM, lower `retrieve_k`/`rerank_k` in `config.py` or switch the
generator to the smaller path (Sarvam-1 stays loaded; skip the 8B fallback).

---

## Colab vs Kaggle (quick contrast)
| | Colab | Kaggle |
|---|---|---|
| Internet | **on by default** | off by default (must toggle) |
| GPU (free) | 1× T4 (16 GB) | 2× T4 (32 GB) or P100 |
| Code in | upload / Drive / git | Dataset / git |
| Secrets | 🔑 panel → `userdata.get` | Add-ons → Secrets |
| Persistence | Google Drive | `/kaggle/working` + Save Version |

---

## Common gotchas
| Symptom | Fix |
|---|---|
| `cuda` not available | Runtime → Change runtime type → **T4 GPU**. |
| Files vanished | Runtime disconnected — use **Drive** (Option B) to persist. |
| Gated model 401/403 | Add `HF_TOKEN` via 🔑 Secrets + accept the model license on HF. |
| Re-downloads each session | Set `HF_HOME` to a Drive path. |
| Session drops mid-run | Free-tier idle/runtime limits; Colab Pro or checkpoint to Drive. |
| Out of memory | Keep 4-bit + on-demand loading; FAISS on CPU; reduce `*_k`. |
