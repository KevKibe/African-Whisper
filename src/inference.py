import torch
import gradio as gr
import pytube as pt
from transformers import pipeline, WhisperTokenizer
from huggingface_hub import model_info
import argparse
import subprocess

try:
    subprocess.run(['sudo', 'apt-get', 'install', 'ffmpeg'], check=True)
    print("FFmpeg installed successfully!")
except subprocess.CalledProcessError as e:
    print(f"Error installing FFmpeg: {e}")
    
def parse_args():
    parser = argparse.ArgumentParser(description="Whisper Demo: Transcribe Audio and YouTube")
    parser.add_argument("--model_name", type=str, help="Name of the fine-tuned model to use in your huggingfacehub repo")
    parser.add_argument("--language_abbr", type=str, help="Language abbreviation for transcription")
    parser.add_argument("--tokenizer", type = str, help = "Whisper model version you used to fine-tune your model e.g openai/whisper-tiny, openai/whisper-base, openai/whisper-small, openai/whisper-medium, openai/whisper-large, openai/whisper-large-v2")
    parser.add_argument("--huggingface_read_token", type = str, help = "Hugging Face API token for read authenticated access.")
    return parser.parse_args()

args = parse_args()


tokenizer = WhisperTokenizer.from_pretrained(args.tokenizer, cache_dir='./ti/tokenizer')

device = 0 if torch.cuda.is_available() else "cpu"
pipe = pipeline(
    task="automatic-speech-recognition",
    model=args.model_name,
    token = args.huggingface_read_token,
    tokenizer=tokenizer, 
    chunk_length_s=30,
    device=device,
)
print(pipe.model.config.model_type)
pipe.model.config.forced_decoder_ids = pipe.tokenizer.get_decoder_prompt_ids(language=args.language_abbr, task="transcribe")

def transcribe(microphone, file_upload):
    warn_output = ""
    if (microphone is not None) and (file_upload is not None):
        warn_output = (
            "WARNING: You've uploaded an audio file and used the microphone. "
            "The recorded file from the microphone will be used and the uploaded audio will be discarded.\n"
        )

    elif (microphone is None) and (file_upload is None):
        return "ERROR: You have to either use the microphone or upload an audio file"

    file = microphone if microphone is not None else file_upload

    text = pipe(file)["text"]

    return warn_output + text


def _return_yt_html_embed(yt_url):
    video_id = yt_url.split("?v=")[-1]
    HTML_str = (
        f'<center> <iframe width="500" height="320" src="https://www.youtube.com/embed/{video_id}"> </iframe>'
        " </center>"
    )
    return HTML_str


def yt_transcribe(yt_url):
    yt = pt.YouTube(yt_url)
    html_embed_str = _return_yt_html_embed(yt_url)
    stream = yt.streams.filter(only_audio=True)[0]
    stream.download(filename="audio.mp3")

    text = pipe("audio.mp3")["text"]

    return html_embed_str, text

demo = gr.Blocks()

mf_transcribe = gr.Interface(
    fn=transcribe,
    inputs=[
        gr.Audio(sources="microphone", type="filepath"),
        gr.Audio(sources="upload", type="filepath"),
    ],
    outputs="text",
    title="Whisper Demo: Transcribe Audio",
    description=(
        "Transcribe long-form microphone or audio inputs with the click of a button! Demo uses the the fine-tuned"
        f" checkpoint [{args.model_name}](https://huggingface.co/{args.model_name}) and 🤗 Transformers to transcribe audio files"
        " of arbitrary length."
    ),
    allow_flagging="never",
)

yt_transcribe = gr.Interface(
    fn=yt_transcribe,
    inputs=[gr.Textbox(lines=1, placeholder="Paste the URL to a YouTube video here", label="YouTube URL")],
    outputs=["html", "text"],
    title="Whisper Demo: Transcribe YouTube",
    description=(
        "Transcribe long-form YouTube videos with the click of a button! Demo uses the the fine-tuned checkpoint:"
        f" [{args.model_name}](https://huggingface.co/{args.model_name}) and 🤗 Transformers to transcribe audio files of"
        " arbitrary length."
    ),
    allow_flagging="never",
)

with demo:
    gr.TabbedInterface([mf_transcribe, yt_transcribe], ["Transcribe Audio", "Transcribe YouTube"])

demo.launch(share = True)
