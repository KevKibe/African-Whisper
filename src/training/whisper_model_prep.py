from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model
import warnings
from transformers import (
    WhisperConfig,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizerFast,
    WhisperTokenizer,

)
import torch
warnings.filterwarnings("ignore")

class WhisperModelPrep:
    """Facilitates the preparation of datasets for fine-tuning with the Whisper model,
    encompassing feature extraction, tokenization, and dataset processing.

    Attributes
    ----------
        dataset (DatasetDict): Dataset to be prepared for the Whisper model.
        model_id (str): Identifier for the Whisper model to be utilized.
        language_code (str): ISO code representing the dataset's language.
        processing_task (str): Specific task for the Whisper model to execute.

    """

    def __init__(
        self,
        model_id: str,
        language: list,
        processing_task: str,
        use_peft: bool,
    ):
        """Sets up the dataset and configuration for processing with the Whisper model.

        Parameters
        ----------
            dataset (DatasetDict): The dataset to be prepared.
            model_id (str, optional): Whisper model identifier.
            language_code (str, optional): Dataset language ISO code.
            processing_task (str, optional): Task for the Whisper model.

        """

        self.model_id = model_id
        self.language = language[0]
        self.processing_task = processing_task
        self.use_peft = use_peft

    def initialize_feature_extractor(self) -> WhisperFeatureExtractor:
        """Creates and retrieves a feature extractor based on the Whisper model.

        Returns
        -------
            WhisperFeatureExtractor: Configured feature extractor for the model.

        """
        return WhisperFeatureExtractor.from_pretrained(
            self.model_id
        )

    def initialize_tokenizer(self) -> WhisperTokenizer:
        """Creates and retrieves a tokenizer for the Whisper model.

        Returns
        -------
            WhisperTokenizer: Configured tokenizer for the model.

        """
        return WhisperTokenizer.from_pretrained(
            self.model_id
        )

    def initialize_processor(self) -> WhisperProcessor:
        """Combines the feature extractor and tokenizer to create a processor for the Whisper model,
        facilitating seamless data preparation.

        Returns
        -------
            WhisperProcessor: Unified processor for the model.

        """
        return WhisperProcessor.from_pretrained(
            self.model_id, self.processing_task
        )

    def initialize_model(self) -> WhisperForConditionalGeneration:
        """Initializes and retrieves the Whisper model configured for conditional generation.

        This method sets up the Whisper model with specific configurations, ensuring it is
        ready for use in tasks such as transcription or translation, depending on the
        processing task specified during the class initialization.

        Returns
        -------
            WhisperForConditionalGeneration: The configured Whisper model ready for conditional generation tasks.

        """
        processor = self.initialize_processor()
        if self.use_peft:
            model = WhisperForConditionalGeneration.from_pretrained(
                self.model_id,
                load_in_8bit=True,
                device_map="auto",
            )
            model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(task=self.processing_task)
            # model.config.suppress_tokens = []
            model.config.use_cache = True
            model.generation_config.language = self.language if self.processing_task == "transcribe" else "en"
            model.generation_config.task = self.processing_task
            model = prepare_model_for_kbit_training(model)
            config = LoraConfig(
                r=32,
                lora_alpha=64,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.05,
                bias="none",
            )
            model = get_peft_model(model, config)
            model.print_trainable_parameters()
        else:
            print("PEFT optimization is not enabled.")
            model = WhisperForConditionalGeneration.from_pretrained(
                self.model_id,
                low_cpu_mem_usage = True
            )
            model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(task=self.processing_task)
            # model.config.suppress_tokens = []
            model.config.use_cache = True
            model.generation_config.language = self.language if self.processing_task == "transcribe" else "en"
            model.generation_config.task = self.processing_task
            model = model.to("cuda") if torch.cuda.is_available() else model
        return model


#################################


def load_model_ps(
        model_name_or_path,
        token,
        model_revision="main",
        cache_dir=None,
        feature_extractor_name=None,
        config_name=None,
        tokenizer_name=None,
        processor_name=None,
        use_fast_tokenizer=True,
        subfolder="",
        attn_implementation=None,
        language=None,
        dtype="bfloat16",
        return_timestamps=False,
        task="transcribe",
):
    if dtype == "float16":
        # mixed_precision = "fp16"
        torch_dtype = torch.float16
    elif dtype == "bfloat16":
        # mixed_precision = "bf16"
        torch_dtype = torch.bfloat16
    else:
        # mixed_precision = "no"
        torch_dtype = torch.float32

    config = WhisperConfig.from_pretrained(
            (config_name if config_name else model_name_or_path),
            cache_dir=cache_dir,
            revision=model_revision,
            token=token,
        )
    feature_extractor = WhisperFeatureExtractor.from_pretrained(
        (feature_extractor_name if feature_extractor_name else model_name_or_path),
        cache_dir=cache_dir,
        revision=model_revision,
        token=token,
    )
    tokenizer = WhisperTokenizerFast.from_pretrained(
        (tokenizer_name if tokenizer_name else model_name_or_path),
        cache_dir=cache_dir,
        use_fast=use_fast_tokenizer,
        revision=model_revision,
        token=token,
    )
    processor = WhisperProcessor.from_pretrained(
        (processor_name if processor_name else model_name_or_path),
        cache_dir=cache_dir,
        revision=model_revision,
        token=token,
    )

    model = WhisperForConditionalGeneration.from_pretrained(
        model_name_or_path,
        config=config,
        cache_dir=cache_dir,
        revision=model_revision,
        subfolder=subfolder,
        token=token,
        low_cpu_mem_usage=True,
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
    )
    model.eval()

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    return_timestamps = return_timestamps
    if hasattr(model.generation_config, "is_multilingual") and model.generation_config.is_multilingual:
        is_multilingual = True
        # We need to set the language and task ids for multilingual checkpoints
        tokenizer.set_prefix_tokens(
            language=language, task=task, predict_timestamps=return_timestamps
        )
    elif language is not None:
        raise ValueError(
            "Setting language token for an English-only checkpoint is not permitted. The language argument should "
            "only be set for multilingual checkpoints."
        )
    else:
        is_multilingual = False
    return model, feature_extractor, tokenizer, processor, is_multilingual
