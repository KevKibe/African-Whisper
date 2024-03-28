import torch
import gradio as gr
import pytube as pt
from transformers import pipeline, WhisperTokenizer
import os


class WhisperDemo:
    def __init__(self, model_name, huggingface_read_token):
        self.model_name = model_name
        self.huggingface_read_token = huggingface_read_token
        self.pipe = None

    def initialize_pipeline(self):
        device = 0 if torch.cuda.is_available() else "cpu"
        self.pipe = pipeline(
            task="automatic-speech-recognition",
            model=self.model_name,
            token=self.huggingface_read_token,
            chunk_length_s=30,
            device=device,
        )

    def transcribe(self, inputs, task):
        if input is None:
            raise gr.Error("No audio file submitted! Please upload or record an audio file before submitting your request.")
        BATCH_SIZE = 8
        text = self.pipe(inputs, batch_size=BATCH_SIZE, generate_kwargs={"task": task}, return_timestamps=True)["text"]
        return  text

    def generate_demo(self):
        self.initialize_pipeline()
        mf_transcribe = gr.Interface(
            fn=self.transcribe,
            inputs=[
                gr.Audio(sources="microphone", type="filepath"),
                gr.Radio(["transcribe", "translate"], label="Task"),
            ],
            outputs="text",
            title="Whisper : Transcribe Audio",
            description=(
                "Transcribe long-form microphone or audio inputs with the click of a button! Demo uses the OpenAI Whisper"
                f" checkpoint [{self.model_name}](https://huggingface.co/{self.model_name}) and 🤗 Transformers to transcribe audio files"
                " of arbitrary length."
            ),
            allow_flagging="never",
            )

        file_transcribe = gr.Interface(
            fn=self.transcribe,
            inputs=[
                gr.Audio(sources="upload", type="filepath", label="Audio file"),
                gr.Radio(["transcribe", "translate"], label="Task"),
            ],
            outputs="text",
            title="Whisper : Transcribe Audio",
            description=(
                "Transcribe long-form microphone or audio inputs with the click of a button! Demo uses the OpenAI Whisper"
                f" checkpoint [{self.model_name}](https://huggingface.co/{self.model_name}) and 🤗 Transformers to transcribe audio files"
                " of arbitrary length."
            ),
            allow_flagging="never",
        )

        demo = gr.TabbedInterface(
            [mf_transcribe, file_transcribe],
            ["Transcribe Audio", "Transcribe YouTube"],
        )
        demo.launch(share=True)
