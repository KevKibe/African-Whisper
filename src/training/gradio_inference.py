import torch
import gradio as gr
from transformers import pipeline
import warnings
import yt_dlp as youtube_dl
import time
from transformers.pipelines.audio_utils import ffmpeg_read
import tempfile
import os
warnings.filterwarnings("ignore")

class WhisperDemo:
    def __init__(self, model_name, huggingface_read_token):
        self.model_name = model_name
        self.huggingface_read_token = huggingface_read_token
        self.pipe = None


    def initialize_pipeline(self):
        device = 0 if torch.cuda.is_available() else "cpu"
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        else:
            dtype = None

        self.pipe = pipeline(
            task="automatic-speech-recognition",
            model=self.model_name,
            token=self.huggingface_read_token,
            device=device,
            torch_dtype=dtype
        )


    def transcribe(self, inputs, task):
        if inputs is None:
            raise gr.Error("No audio file submitted! Please upload or record an audio file before submitting your request.")
        text = self.pipe(
                        inputs,
                        chunk_length_s=30,
                        batch_size=24,
                        return_timestamps=True,
                        generate_kwargs={"task": task}
                        )["text"]
        return  text

    def _return_yt_html_embed(self, yt_url):
        video_id = yt_url.split("?v=")[-1]
        HTML_str = (
            f'<center> <iframe width="500" height="320" src="https://www.youtube.com/embed/{video_id}"> </iframe>'
            " </center>"
        )
        return HTML_str

    def download_yt_audio(self, yt_url, filename):
        YT_LENGTH_LIMIT_S = 3600
        info_loader = youtube_dl.YoutubeDL()

        try:
            info = info_loader.extract_info(yt_url, download=False)
        except youtube_dl.utils.DownloadError as err:
            raise gr.Error(str(err))

        file_length = info["duration_string"]
        file_h_m_s = file_length.split(":")
        file_h_m_s = [int(sub_length) for sub_length in file_h_m_s]

        if len(file_h_m_s) == 1:
            file_h_m_s.insert(0, 0)
        if len(file_h_m_s) == 2:
            file_h_m_s.insert(0, 0)
        file_length_s = file_h_m_s[0] * 3600 + file_h_m_s[1] * 60 + file_h_m_s[2]

        if file_length_s > YT_LENGTH_LIMIT_S:
            yt_length_limit_hms = time.strftime("%HH:%MM:%SS", time.gmtime(YT_LENGTH_LIMIT_S))
            file_length_hms = time.strftime("%HH:%MM:%SS", time.gmtime(file_length_s))
            raise gr.Error(f"Maximum YouTube length is {yt_length_limit_hms}, got {file_length_hms} YouTube video.")

        ydl_opts = {"outtmpl": filename, "format": "worstvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best"}

        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            try:
                ydl.download([yt_url])
            except youtube_dl.utils.ExtractorError as err:
                raise gr.Error(str(err))

    def yt_transcribe(self, yt_url, task, max_filesize=75.0):
        BATCH_SIZE = 8
        html_embed_str = self._return_yt_html_embed(yt_url)

        with tempfile.TemporaryDirectory() as tmpdirname:
            filepath = os.path.join(tmpdirname, "video.mp4")
            self.download_yt_audio(yt_url, filepath)
            with open(filepath, "rb") as f:
                inputs = f.read()

        inputs = ffmpeg_read(inputs, self.pipe.feature_extractor.sampling_rate)
        inputs = {"array": inputs, "sampling_rate": self.pipe.feature_extractor.sampling_rate}

        text = self.pipe(inputs, batch_size=BATCH_SIZE, generate_kwargs={"task": task}, return_timestamps=True)["text"]

        return html_embed_str, text

    def generate_demo(self):
        self.initialize_pipeline()

        mf_transcribe = gr.Interface(
            fn=self.transcribe,
            inputs=[
                gr.Audio(sources="microphone", type="filepath"),
                gr.Radio(["transcribe", "translate"], label="Task"),
            ],
            outputs="text",
            title="Transcribe Audio",
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
            title="Transcribe Audio",
            description=(
                "Transcribe long-form microphone or audio inputs with the click of a button! Demo uses the OpenAI Whisper"
                f" checkpoint [{self.model_name}](https://huggingface.co/{self.model_name}) and 🤗 Transformers to transcribe audio files"
                " of arbitrary length."
            ),
            allow_flagging="never",
        )

        yt_transcribe = gr.Interface(
            fn=self.yt_transcribe,
            inputs=[
                gr.Textbox(lines=1, placeholder="Paste the URL to a YouTube video here", label="YouTube URL"),
                gr.Radio(["transcribe", "translate"], label="Task")
            ],
            outputs=["html", "text"],
            title="Transcribe YouTube",
            description=(
                "Transcribe long-form YouTube videos with the click of a button! Demo uses the OpenAI Whisper checkpoint"
                f" [{self.model_name}](https://huggingface.co/{self.model_name}) and 🤗 Transformers to transcribe video files of"
                " arbitrary length. The duration of this task will vary depending on the length of the video."
            ),
            allow_flagging="never",
            )

        demo = gr.TabbedInterface(
            [mf_transcribe, file_transcribe, yt_transcribe],
            ["Transcribe Audio", "Transcribe mp3 File", "Transcribe YouTube Video"],
        )
        demo.launch(share=True, debug=True)
