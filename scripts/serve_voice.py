"""Gradio voice demo for SETU-RAG speech-to-speech (turn-based).

    python scripts/serve_voice.py     # opens a mic-in / audio-out chat
"""
def main():
    import gradio as gr
    # from setu_rag.pipeline import SetuRAG
    # from setu_rag.speech_pipeline import SpeechSetuRAG
    # from setu_rag.speech.audio_io import from_array
    # rag = SetuRAG(index=..., translator=...)        # build the text core
    # voice = SpeechSetuRAG(rag)

    def respond(mic_path):
        # turn = voice.answer_file(mic_path, "reply.wav")
        # return "reply.wav", turn.transcript + "  ->  " + turn.answer_text
        return None, "Build the index + SETU-RAG core, then this returns spoken replies."

    with gr.Blocks(title="SETU-RAG — Voice (code-switch-aware)") as demo:
        gr.Markdown("## SETU-RAG — speak in any of 22 Indian languages + English")
        mic = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Speak")
        out_audio = gr.Audio(label="Spoken reply", autoplay=True)
        out_text = gr.Textbox(label="Transcript -> answer")
        mic.change(respond, inputs=mic, outputs=[out_audio, out_text])
    demo.launch()

if __name__ == "__main__":
    main()
