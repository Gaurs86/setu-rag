# Prompt for Claude Code

Paste the block below into Claude Code (the "code tab") with this repo attached.

---

You are working on SETU-RAG, my MTech CS dissertation project: a code-switching-aware
multilingual SPEECH-TO-SPEECH RAG assistant for customer support, covering the 22 official
Indian languages mixed with English, designed to run on a single Google Colab T4 (16 GB).
Input and output are audio; the text core is a novel RAG pipeline. The repo is attached.

FIRST, read these to understand the system before changing anything:
README.md, COLAB.md, setu_rag/config.py, setu_rag/app.py, setu_rag/pipeline.py,
setu_rag/speech_pipeline.py (and the SETU-RAG_*_Architecture docs if present).

ARCHITECTURE (already implemented)
Text core (setu_rag/): front-end LID + Code-Mixing-Index (CMI) -> CMI-adaptive retrieval
router [novel] -> transliteration-robust multi-view query [novel] -> hybrid dense (BGE-M3)
+ BM25 retrieval with Reciprocal Rank Fusion -> cross-encoder rerank (BGE-reranker-v2-m3)
-> CRAG corrective grading -> CMI-conditioned generation [novel] -> faithfulness gate.
Speech layer (setu_rag/speech/, one file per stage): VAD -> ASR -> spoken-LID ->
acoustic+lexical LID fusion [novel] -> text core -> CMI-conditioned TTS [novel] -> audio.
Entry point: `from setu_rag.app import build_pipeline; rag = build_pipeline();
rag.answer("mera refund kab aayega")`.

CURRENT STATE
- Every model wrapper has a REAL implementation + a graceful FALLBACK, so it runs on Colab
  GPU with real models and offline/CPU with stand-ins. NEVER remove the fallbacks.
- Real & un-gated defaults (no HF token needed): embedder BAAI/bge-m3, reranker
  BAAI/bge-reranker-v2-m3, generator Qwen/Qwen2.5-3B-Instruct (4-bit).
- Still heuristic/optional: front-end LID is a script+cue heuristic (real ai4bharat/IndicLID
  is a drop-in); transliteration (IndicXlit) and translation (IndicTrans2) are optional and
  currently identity/skipped, so the native-script and English-pivot query views are limited
  until wired.
- Dissertation model choices (sarvamai/sarvam-1, CohereForAI/aya-expanse-8b,
  ai4bharat/indic-conformer-600m-multilingual, ai4bharat/indic-parler-tts) live in config.py;
  the demo defaults to un-gated models. Keep BOTH documented.

HOW TO RUN / TEST
  pip install -r colab_requirements.txt
  python scripts/build_index.py --offline   # structure check, no downloads, no GPU
  python scripts/build_index.py             # real models (GPU)
After ANY change, run `python scripts/build_index.py --offline` and confirm it still answers
without errors. force_offline=True must keep working everywhere (it is my no-GPU test path).

WHAT I WANT YOU TO DO (propose a short plan first, then work incrementally)
1. Confirm it runs (offline smoke test) and fix anything broken.
2. Wire the optional AI4Bharat components as real-with-fallback, ONE AT A TIME, testing after
   each: (a) language_id.py -> real IndicLID; (b) transliterate.py -> IndicXlit; (c) a real
   IndicTrans2 translator passed into build_pipeline so the multi-view English-pivot /
   matrix-canonical views work; (d) speech/asr.py -> IndicConformer primary, Whisper fallback.
3. Add a small real evaluation: a handful of code-switched QA pairs run through setu_rag/eval/
   (CS-RAGAS + WER/CER) printing a results table.
4. Stay within T4 memory: 4-bit generation, on-demand loading + freeing, FAISS on CPU.

CONVENTIONS
- One Python file per pipeline stage; keep the module layout.
- Preserve the lazy-load + graceful-fallback pattern in every model wrapper.
- Update README.md / COLAB.md and the relevant notebook when behavior changes.
- After each change: run the offline smoke test, then commit with a clear message
  (e.g. "feat(asr): wire IndicConformer with Whisper fallback").

Begin by reading the files above and running the offline smoke test, then propose your plan.
