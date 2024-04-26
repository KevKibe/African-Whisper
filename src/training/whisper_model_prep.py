from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor
from transformers import WhisperForConditionalGeneration
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model
import warnings

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

        if self.use_peft:
            model = WhisperForConditionalGeneration.from_pretrained(
                self.model_id,
                load_in_8bit=True,
                device_map="auto",
            )
            model.config.forced_decoder_ids = None
            model.config.suppress_tokens = []
            model.generation_config.language = "en"
            model.generation_config.task = "translate"
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
                self.model_id
            )
            model.config.forced_decoder_ids = None
            model.config.suppress_tokens = []
            model.generation_config.language = "en"
            model.generation_config.task = "translate"

        return model
