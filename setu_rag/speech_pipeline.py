"""Speech-to-speech orchestration: VANI layer wrapping the SETU-RAG text core.

    audio in
      -> VAD segment
      -> [per segment]  ASR  +  spoken-LID
      -> acoustic+lexical LID fusion        [speech novelty #1]
      -> assemble transcript + matrix language
      -> SETU-RAG text core (SetuRAG.answer) -> grounded answer text
      -> CMI-conditioned TTS                [speech novelty #2]
      -> audio out

Turn-based now, streaming-ready: `answer_audio` consumes a whole clip; the VAD and
ASR stages already expose chunk-level hooks so a live front-end can stream later.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
from .config import SETTINGS
from .speech.audio_io import Audio, load as load_audio, save as save_audio
from .speech.vad import VAD
from .speech.asr import ASR
from .speech.spoken_lid import SpokenLID
from .speech.lid_fusion import fuse
from .speech.tts import TTS
from .front_end.language_id import LanguageIdentifier
from .pipeline import SetuRAG, Trace

@dataclass
class SpeechTurn:
    transcript: str = ""
    matrix_lang: str = ""
    cmi: float = 0.0
    answer_text: str = ""
    answer_audio: Optional[Audio] = None
    text_trace: Optional[Trace] = None
    timings_ms: dict = field(default_factory=dict)

class SpeechSetuRAG:
    def __init__(self, rag: SetuRAG, settings=SETTINGS):
        self.s = settings
        self.rag = rag                       # the Round-1 text pipeline
        self.vad = VAD()
        self.asr = ASR(device=settings.device, force_offline=settings.force_offline)
        self.slid = SpokenLID(device=settings.device)
        self.tts = TTS(device=settings.device, force_offline=settings.force_offline)
        self.lid = LanguageIdentifier(device=settings.device, force_offline=settings.force_offline)

    def answer_audio(self, audio: Audio) -> SpeechTurn:
        import time
        turn = SpeechTurn()

        # 1) segment
        t0 = time.time()
        segments = self.vad.segment(audio)

        # 2) ASR + spoken-LID per segment  (models loaded on demand to fit T4)
        texts, acoustic = [], {}
        for seg in segments:
            ac = self.slid.identify(seg.audio)
            hint = max(ac, key=ac.get) if ac else None
            res = self.asr.transcribe(seg.audio, lang_hint=hint)
            texts.append(res.text)
            for k, v in ac.items():
                acoustic[k] = acoustic.get(k, 0.0) + v
        transcript = " ".join(t for t in texts if t).strip()
        # normalise acoustic posterior
        tot = sum(acoustic.values()) or 1.0
        acoustic = {k: v / tot for k, v in acoustic.items()}
        turn.timings_ms["asr"] = int((time.time() - t0) * 1000)

        # 3) LID fusion (acoustic prior + lexical tags) -> robust matrix language
        tokens = self.lid.tag(transcript)
        fused = fuse(acoustic, tokens, alpha=self.s.lid_fusion_alpha)
        turn.transcript = transcript

        # 4) text core. The fused matrix language can be injected to override the
        #    core's text-only LID (integration hook); here we run the core and read
        #    back its CMI/matrix-lang, preferring the fused matrix language for TTS.
        t1 = time.time()
        tr = self.rag.answer(transcript)
        turn.text_trace = tr
        turn.matrix_lang = fused.matrix_lang or tr.matrix_lang
        turn.cmi = tr.cmi
        turn.answer_text = tr.answer
        turn.timings_ms["rag"] = int((time.time() - t1) * 1000)

        # 5) CMI-conditioned speech synthesis
        t2 = time.time()
        turn.answer_audio = self.tts.synthesize(turn.answer_text, turn.matrix_lang, turn.cmi)
        turn.timings_ms["tts"] = int((time.time() - t2) * 1000)
        return turn

    # convenience file-in/file-out
    def answer_file(self, in_path: str, out_path: str) -> SpeechTurn:
        turn = self.answer_audio(load_audio(in_path))
        if turn.answer_audio is not None:
            save_audio(out_path, turn.answer_audio)
        return turn
