"""CMI-Conditioned Generation  ---  NOVEL CONTRIBUTION #3 (now runnable).

Live path: a causal LM (default Qwen2.5-3B-Instruct — un-gated, multilingual,
T4-friendly) generates an answer grounded in the retrieved context, with a system
prompt conditioned on the user's matrix language and Code-Mixing Index so the reply
mirrors their register. Load strategy: try 4-bit (bitsandbytes NF4) → fp16 → fp32.
Offline fallback: an extractive answer (the top retrieved passage).
"""
from __future__ import annotations
from typing import List
from ..config import SETTINGS, SARVAM1_LANGS

SYSTEM = ("You are a helpful customer-support assistant. Answer ONLY using the provided context. "
          "If the context does not contain the answer, say so briefly and offer a human agent. "
          "Keep the answer short.")

def _style_directive(matrix_lang: str, cmi: float) -> str:
    fam = matrix_lang.split("_")[0]
    if cmi < 0.05:
        return f"Reply entirely in {fam}."
    if cmi < 0.20:
        return f"Reply mainly in {fam} with a few English technical words, lightly code-mixed."
    return (f"Reply in a natural {fam}-English code-mixed style (about CMI {cmi:.2f}), "
            f"keeping {fam} as the grammatical frame.")

def build_prompt(query: str, contexts: List[dict], matrix_lang: str, cmi: float) -> str:
    ctx = "\n\n".join(f"[{i+1}] {c.get('answer','')}" for i, c in enumerate(contexts))
    return (f"{SYSTEM}\n{_style_directive(matrix_lang, cmi)}\n\n"
            f"Context:\n{ctx}\n\nUser: {query}\nAssistant:")

class Generator:
    def __init__(self, settings=SETTINGS):
        self.s = settings
        which = "gen_demo" if settings.prefer_demo_generator else "gen_primary"
        self.model_id = settings.models.get(which) or settings.models["gen_primary"]
        self._tok = None
        self._model = None
        self.live = False

    def load(self):
        if self._model is not None:
            return self
        if self.s.force_offline:
            self._model = "extractive"; self.live = False; return self

        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        try:
            self._tok = AutoTokenizer.from_pretrained(self.model_id)
        except Exception as e:
            print(f"[generator] tokenizer load failed ({type(e).__name__}: {e}); "
                  "using extractive fallback.")
            self._model = "extractive"; self.live = False; return self

        # Try loading in order: 4-bit → fp16 → fp32
        for attempt, label, kw in self._load_strategies(torch):
            try:
                self._model = AutoModelForCausalLM.from_pretrained(self.model_id, **kw)
                self.live = True
                print(f"[generator] loaded {self.model_id} ({label})")
                return self
            except Exception as e:
                print(f"[generator] {label} load failed ({type(e).__name__}: {e}); "
                      f"{'trying next strategy' if attempt < 2 else 'using extractive fallback'}.")

        self._model = "extractive"; self.live = False
        return self

    def _load_strategies(self, torch):
        """Yield (attempt_index, label, kwargs) from most to least efficient."""
        base = {"device_map": "auto"}
        # 4-bit (requires bitsandbytes)
        if self.s.load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig
                bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                        bnb_4bit_compute_dtype=torch.float16)
                yield 0, "4-bit NF4", {**base, "quantization_config": bnb}
            except Exception as e:
                print(f"[generator] BitsAndBytesConfig unavailable ({type(e).__name__}: {e}).")
        # fp16
        yield 1, "fp16", {**base, "torch_dtype": torch.float16}
        # fp32 (last resort — may OOM on T4 for large models)
        yield 2, "fp32", base

    def generate(self, query: str, contexts: List[dict], matrix_lang: str, cmi: float) -> str:
        self.load()
        if not self.live:
            return self._extractive(contexts)
        sys_d = SYSTEM + " " + _style_directive(matrix_lang, cmi)
        ctx = "\n\n".join(f"[{i+1}] {c.get('answer','')}" for i, c in enumerate(contexts))
        user = f"Context:\n{ctx}\n\nQuestion: {query}"
        try:
            prompt = self._tok.apply_chat_template(
                [{"role": "system", "content": sys_d}, {"role": "user", "content": user}],
                add_generation_prompt=True, tokenize=False)
        except Exception:
            prompt = build_prompt(query, contexts, matrix_lang, cmi)
        enc = self._tok(prompt, return_tensors="pt").to(self._model.device)
        out = self._model.generate(**enc, max_new_tokens=256, do_sample=False,
                                   pad_token_id=(self._tok.eos_token_id or
                                                 self._tok.pad_token_id or 0))
        return self._tok.decode(out[0][enc["input_ids"].shape[1]:],
                                skip_special_tokens=True).strip()

    @staticmethod
    def _extractive(contexts: List[dict]) -> str:
        if not contexts:
            return "Sorry, I couldn't find this in our help centre — connecting you to an agent."
        return contexts[0].get("answer", "")
