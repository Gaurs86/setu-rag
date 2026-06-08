"""Gradio voice demo for SETU-RAG speech-to-speech (turn-based).

    python scripts/serve_voice.py             # real models if available, fallbacks otherwise
    python scripts/serve_voice.py --offline   # structure check (empty transcript, no audio out)
"""
import argparse
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--faqs", default=None)
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()

    import gradio as gr
    import numpy as np
    from setu_rag.app import build_pipeline
    from setu_rag.speech_pipeline import SpeechSetuRAG
    from setu_rag.speech.audio_io import from_array

    rag = build_pipeline(faq_path=args.faqs, force_offline=args.offline)
    voice = SpeechSetuRAG(rag)

    def respond(mic):
        if mic is None:
            return None, "Speak into the mic (or upload a clip)."
        sr, samples = mic
        samples = np.asarray(samples, dtype="float32")
        if samples.ndim == 2:                      # stereo -> mono
            samples = samples.mean(axis=1)
        if samples.size and np.abs(samples).max() > 1.0:   # int16 -> [-1, 1]
            samples = samples / 32768.0
        turn = voice.answer_audio(from_array(samples, sr))
        out_audio = None
        if turn.answer_audio is not None:
            out_audio = (turn.answer_audio.sr, turn.answer_audio.samples)
        return out_audio, f"{turn.transcript or '(no transcript)'}  ->  {turn.answer_text}"

    with gr.Blocks(title="SETU-RAG — Voice (code-switch-aware)") as demo:
        gr.Markdown("## SETU-RAG — speak in any of 22 Indian languages + English")
        mic = gr.Audio(sources=["microphone", "upload"], type="numpy", label="Speak")
        out_audio = gr.Audio(label="Spoken reply", autoplay=True)
        out_text = gr.Textbox(label="Transcript -> answer")
        mic.change(respond, inputs=mic, outputs=[out_audio, out_text])
    demo.launch(share=args.share)


if __name__ == "__main__":
    main()
