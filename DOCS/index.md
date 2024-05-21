<h1 align="center">African Whisper: ASR for African Languages</h1>

<p align="center">
  <a href="https://twitter.com/AfriWhisper">
    <img src="https://img.shields.io/twitter/follow/AfriWhisper?style=social" alt="Twitter">
  </a>
  <a href="https://github.com/KevKibe/African-Whisper/commits/">
    <img src="https://img.shields.io/github/last-commit/KevKibe/African-Whisper?" alt="Last commit">
  </a>
  <a href="https://github.com/KevKibe/African-Whisper/blob/main/LICENSE">
    <img src="https://img.shields.io/github/license/KevKibe/African-Whisper?" alt="License">
  </a>

</p>

<p align="center">
    <img src= "logo_image.png" width="100">
</p>


*Framework for seamless fine-tuning and deploying Whisper Model developed to advance Automatic Speech Recognition (ASR): translation and transcription capabilities for African languages*.


## Features
  
- 🔧 **Fine-Tuning**: Fine-tune the [Whisper](https://huggingface.co/collections/openai/whisper-release-6501bba2cf999715fd953013) model on any audio dataset from Huggingface, e.g., [Mozilla's](https://huggingface.co/mozilla-foundation) Common Voice datasets.

- 📊 **Metrics Monitoring**: View training run metrics on [Wandb](https://wandb.ai/).

- 🐳 **Production Deployment**: Seamlessly containerize and deploy the model inference endpoint for real-world applications.

- 🚀 **Model Optimization**: Utilize CTranslate2 for efficient model optimization, ensuring faster inference times.

- 📝 **Word-Level Transcriptions**: Produce detailed word-level transcriptions and translations, complete with timestamps.

- 🎙️ **Multi-Speaker Diarization**: Perform speaker identification and separation in multi-speaker audio using diarization techniques.

- 🔍 **Alignment Precision**: Improve transcription and translation accuracy by aligning outputs with Wav2vec models.

- 🛡️ **Reduced Hallucination**: Leverage Voice Activity Detection (VAD) to minimize hallucination and improve transcription clarity.
<br>
The framework implements the following papers:
<br>

1. [Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) : Speech processing systems trained to predict large amounts of transcripts of audio on the internet scaled to 680,000 hours of multilingual and multitask supervision.

2. [WhisperX](https://arxiv.org/abs/2303.00747): Time-Accurate Speech Transcription of Long-Form Audio for time-accurate speech recognition with word-level timestamps. 

3. [Pyannote.audio](https://arxiv.org/abs/1911.01255): Neural building blocks for speaker diarization for advanced speaker diarization capabilities. 

4. [Efficient and High-Quality Neural Machine Translation with OpenNMT](https://arxiv.org/abs/1701.02810): Efficient neural machine translation and model acceleration.  

For more details, you can refer to the [Whisper ASR model paper](https://cdn.openai.com/papers/whisper.pdf).<br>

## Documentation
Refer to the [Documentation](https://kevkibe.github.io/African-Whisper/gettingstarted/) to get started


## Contributing 
Contributions are welcome and encouraged.

Before contributing, please take a moment to review our [Contribution Guidelines](https://github.com/KevKibe/African-Whisper/blob/master/DOCS/CONTRIBUTING.md) for important information on how to contribute to this project.

If you're unsure about anything or need assistance, don't hesitate to reach out to us or open an issue to discuss your ideas.

We look forward to your contributions!


## License
This project is licensed under the MIT License - see the [LICENSE](https://github.com/KevKibe/African-Whisper/blob/main/LICENSE) file for details.

## Contact
For any enquiries, please reach out to me through keviinkibe@gmail.com
