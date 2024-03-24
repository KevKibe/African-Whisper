import torch
import gradio as gr
import pytube as pt
from transformers import pipeline, WhisperTokenizer

class WhisperDemo:
    def __init__(self, model_name, language_abbr, tokenizer, huggingface_read_token):
        self.model_name = model_name
        self.language_abbr = language_abbr
        self.tokenizer = tokenizer
        self.huggingface_read_token = huggingface_read_token
        self.pipe = None

    def initialize_pipeline(self):
        tokenizer = WhisperTokenizer.from_pretrained(self.tokenizer)
        device = 0 if torch.cuda.is_available() else "cpu"
        self.pipe = pipeline(
            task="automatic-speech-recognition",
            model=self.model_name,
            token=self.huggingface_read_token,
            tokenizer=tokenizer,
            chunk_length_s=30,
            device=device,
        )
        self.pipe.model.config.forced_decoder_ids = self.pipe.tokenizer.get_decoder_prompt_ids(language=self.language_abbr, task="transcribe")

    def transcribe(self, microphone, file_upload):
        warn_output = ""
        if (microphone is not None) and (file_upload is not None):
            warn_output = (
                "WARNING: You've uploaded an audio file and used the microphone. "
                "The recorded file from the microphone will be used and the uploaded audio will be discarded.\n"
            )

        elif (microphone is None) and (file_upload is None):
            return "ERROR: You have to either use the microphone or upload an audio file"

        file = microphone if microphone is not None else file_upload
        text = self.pipe(file)["text"]

        return warn_output + text

    def yt_transcribe(self, yt_url):
        yt = pt.YouTube(yt_url)
        video_id = yt_url.split("?v=")[-1]
        html_embed_str = (
            f'<center> <iframe width="500" height="320" src="https://www.youtube.com/embed/{video_id}"> </iframe>'
            " </center>"
        )
        stream = yt.streams.filter(only_audio=True)[0]
        stream.download(filename="audio.mp3")
        text = self.pipe("audio.mp3")["text"]

        return html_embed_str, text

    def generate_demo(self):
        self.initialize_pipeline()

        mf_transcribe = gr.Interface(
            fn=self.transcribe,
            inputs=[
                gr.Audio(sources="microphone", type="filepath"),
                gr.Audio(sources="upload", type="filepath"),
            ],
            outputs="text",
            title="Whisper Demo: Transcribe Audio",
            description=(
                "Transcribe long-form microphone or audio inputs with the click of a button! Demo uses the fine-tuned"
                f" checkpoint [{self.model_name}](https://huggingface.co/{self.model_name}) and 🤗 Transformers to transcribe audio files"
                " of arbitrary length."
            ),
            allow_flagging="never",
        )

        yt_transcribe_interface = gr.Interface(
            fn=self.yt_transcribe,
            inputs=[gr.Textbox(lines=1, placeholder="Paste the URL to a YouTube video here", label="YouTube URL")],
            outputs=["html", "text"],
            title="Whisper Demo: Transcribe YouTube",
            description=(
                "Transcribe long-form YouTube videos with the click of a button! Demo uses the fine-tuned checkpoint:"
                f" [{self.model_name}](https://huggingface.co/{self.model_name}) and 🤗 Transformers to transcribe audio files of"
                " arbitrary length."
            ),
            allow_flagging="never",
        )

        demo = gr.TabbedInterface([mf_transcribe, yt_transcribe_interface], ["Transcribe Audio", "Transcribe YouTube"])
        demo.launch(share=True)
