"""Central configuration: model IDs, language tables, and T4 memory policy.

Every model named here is open-source and was selected to fit a single
Google Colab T4 (16 GB VRAM). Heavy models are loaded on demand and freed
after use (see `setu_rag.utils.model_manager` in your implementation).
"""
from dataclasses import dataclass, field
from typing import Dict, List

# ----- Open-source model registry (HF Hub IDs) -------------------------------
MODELS: Dict[str, str] = {
    # Front-end
    "lid":            "ai4bharat/IndicLID",            # native + romanized LID, 22 langs
    "translit":       "ai4bharat/indicxlit",          # roman <-> native, 21 langs (pip: ai4bharat-transliteration)
    # Retrieval / indexing
    "embedder":       "BAAI/bge-m3",                   # dense + sparse + colbert, 100+ langs, MIT
    "reranker":       "BAAI/bge-reranker-v2-m3",       # multilingual cross-encoder
    # Translation (query expansion + KB pivoting), distilled for T4
    "mt_en_indic":    "ai4bharat/indictrans2-en-indic-dist-200M",
    "mt_indic_en":    "ai4bharat/indictrans2-indic-en-dist-200M",
    "mt_indic_indic": "ai4bharat/indictrans2-indic-indic-dist-320M",
    # Generation (primary: Indic-specialised 2B; fallback: 8B multilingual)
    "gen_primary":    "sarvamai/sarvam-1",             # 2B, 10 Indic langs, efficient tokenizer
    "gen_fallback":   "CohereForAI/aya-expanse-8b",    # 23 langs incl. low-resource Indic; 4-bit on T4
    # Un-gated, multilingual, instruction-tuned default used for the *runnable* Colab demo
    # (no HF token needed). Swap to gen_primary/gen_fallback for the dissertation experiments.
    "gen_demo":       "Qwen/Qwen2.5-3B-Instruct",      # ~2 GB in 4-bit on T4, good Hindi/Indic
    # Optional faithfulness NLI judge (small)
    "nli":            "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
    # ----- Speech I/O layer (VANI) -----
    "vad":            "snakers4/silero-vad",                          # tiny CPU VAD
    "asr":            "ai4bharat/indic-conformer-600m-multilingual",  # 22 langs, code-switched (IndicVoices)
    "asr_fallback":   "openai/whisper-large-v3-turbo",               # general baseline / fallback
    "spoken_lid":     "ai4bharat/indic-conformer-600m-multilingual", # acoustic LID head
    "tts":            "ai4bharat/indic-parler-tts",                  # 21 langs, code-mixing, description-controllable
    "tts_hq":         "ai4bharat/IndicF5",                           # higher naturalness, 11 langs
}

# 22 scheduled languages of India (ISO 639 codes used by IndicTrans2 / FLORES)
SCHEDULED_LANGS: List[str] = [
    "asm_Beng","ben_Beng","brx_Deva","doi_Deva","guj_Gujr","hin_Deva","kan_Knda",
    "kas_Arab","kok_Deva","mai_Deva","mal_Mlym","mni_Mtei","mar_Deva","npi_Deva",
    "ory_Orya","pan_Guru","san_Deva","sat_Olck","snd_Deva","tam_Taml","tel_Telu","urd_Arab",
]
# Subset Sarvam-1 covers natively; others route to the fallback generator.
SARVAM1_LANGS = {"ben_Beng","guj_Gujr","hin_Deva","kan_Knda","mal_Mlym",
                 "mar_Deva","ory_Orya","pan_Guru","tam_Taml","tel_Telu"}

@dataclass
class Settings:
    device: str = "cuda"
    load_in_4bit: bool = True          # bitsandbytes NF4 for the generator
    max_ctx_docs: int = 5              # docs passed to the generator after rerank
    retrieve_k: int = 30               # candidates per view before fusion
    rerank_k: int = 8                  # kept after cross-encoder
    rrf_k: int = 60                    # RRF constant
    cmi_high: float = 0.20             # >= -> treat query as genuinely code-mixed
    cmi_low: float = 0.05              # <  -> treat as effectively monolingual
    faithfulness_threshold: float = 0.7
    # Speech layer
    target_sr: int = 16000             # all speech models expect 16 kHz mono
    lid_fusion_alpha: float = 0.4      # weight on the acoustic prior in LID fusion
    asr_conf_fallback: float = 0.55    # below this ASR confidence, try Whisper fallback
    # Runtime knobs
    prefer_demo_generator: bool = True # use gen_demo (un-gated) instead of Sarvam-1 for live runs
    force_offline: bool = False        # skip model downloads; use deterministic fallbacks everywhere
    models: Dict[str, str] = field(default_factory=lambda: dict(MODELS))

SETTINGS = Settings()
