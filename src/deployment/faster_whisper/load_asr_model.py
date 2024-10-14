from .asr import WhisperModel, FasterWhisperPipeline
from typing import Optional
import faster_whisper
import torch
from .vad import load_vad_model



def load_asr_model(whisper_arch,
               device,
               device_index=0,
               compute_type="float16",
               asr_options=None,
               language : Optional[str] = None,
               vad_model=None,
               vad_options=None,
               model : Optional[WhisperModel] = None,
               download_root=None,
               threads=4,
               is_v3_architecture=False):
    """
    Loads and returns a Whisper ASR model for inference.

    Args:
        whisper_arch (str): Name of the Whisper model architecture to load.
        device (str): Device to load the model on (e.g., "cuda" or "cpu").
        device_index (int): Index of the device, if applicable. Default is 0.
        compute_type (str): Type of compute precision to use (e.g., "float16").
        asr_options (dict, optional): Additional ASR-specific options.
        language (str, optional): Language code for the model. Defaults to None.
        vad_model (optional): Voice Activity Detection model instance, if used.
        vad_options (dict, optional): Options for VAD, if applicable.
        model (WhisperModel, optional): Preloaded WhisperModel instance, if available.
        download_root (str, optional): Directory path for downloading model files.
        threads (int): Number of CPU threads per worker. Default is 4.
        is_v3_architecture (bool): Indicates whether the model uses the v3 architecture.

    Returns:
        object: The Whisper model pipeline for ASR inference.
    """

    if whisper_arch.endswith(".en"):
        language = "en"

    model = model or WhisperModel(whisper_arch,
                         device=device,
                         device_index=device_index,
                         compute_type=compute_type,
                         download_root=download_root,
                         cpu_threads=threads)

    if is_v3_architecture:
        model.feature_extractor.mel_filters = model.feature_extractor.get_mel_filters(
            model.feature_extractor.sampling_rate,
            model.feature_extractor.n_fft,
            n_mels=128
        )
    else:
        model.feature_extractor.mel_filters = model.feature_extractor.get_mel_filters(
            model.feature_extractor.sampling_rate,
            model.feature_extractor.n_fft,
            n_mels=80
        )

    if language is not None:
        tokenizer = faster_whisper.tokenizer.Tokenizer(model.hf_tokenizer, model.model.is_multilingual, language=language)
    else:
        print("No language specified, language will be first be detected for each audio file (increases inference time).")
        tokenizer = None

    default_asr_options =  { # explore temperature_increment_on_fallback parameter
        "beam_size": 5,
        "best_of": 5,
        "patience": 1,
        "length_penalty": 1,
        "repetition_penalty": 1,
        "no_repeat_ngram_size": 0,
        "temperatures": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        "compression_ratio_threshold": 2.4,
        "log_prob_threshold": -1.0,
        "no_speech_threshold": 0.6,
        "condition_on_previous_text": False, # explore True
        "prompt_reset_on_temperature": 0.5,
        "initial_prompt": None,
        "prefix": None,
        "suppress_blank": True,
        "suppress_tokens": [-1],
        "without_timestamps": True,
        "max_initial_timestamp": 0.0,
        "word_timestamps": False, # Explore True
        "prepend_punctuations": "\"'“¿([{-",
        "append_punctuations": "\"'.。,，!！?？:：”)]}、",
        "suppress_numerals": False,
        "max_new_tokens": None,
        "clip_timestamps": None,
        "hallucination_silence_threshold": None,
        "hotwords": None
    }

    if asr_options is not None:
        default_asr_options.update(asr_options)

    suppress_numerals = default_asr_options["suppress_numerals"]
    del default_asr_options["suppress_numerals"]

    default_asr_options = faster_whisper.transcribe.TranscriptionOptions(**default_asr_options)

    default_vad_options = {
        "vad_onset": 0.500,
        "vad_offset": 0.363
    }

    if vad_options is not None:
        default_vad_options.update(vad_options)

    if vad_model is not None:
        vad_model = vad_model
    else:
        vad_model = load_vad_model(torch.device(device), use_auth_token=None, **default_vad_options)

    return FasterWhisperPipeline(
        model=model,
        vad=vad_model,
        options=default_asr_options,
        tokenizer=tokenizer,
        language=language,
        suppress_numerals=suppress_numerals,
        vad_params=default_vad_options,
        is_v3_architecture=is_v3_architecture
    )
