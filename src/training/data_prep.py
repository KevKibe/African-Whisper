from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
)
from .load_data import Dataset
from .whisper_model_prep import WhisperModelPrep
from .audio_data_processor import AudioDataProcessor
from datasets import DatasetDict
from typing import Tuple
import warnings
warnings.filterwarnings("ignore")

class DataPrep:
    """

    A class to encapsulate preparing the speech dataset for model training, including dataset loading, cleaning, and preprocessing tasks like feature extraction and tokenization. .

    """

    def __init__(
        self,
        huggingface_read_token: str,
        dataset_name: str,
        language_abbr: str,
        model_id: str,
        processing_task: str,
        use_peft: bool,
    ):
        """
        Initializes the Trainer with the necessary configuration and loads the evaluation metric.

        Parameters:
            huggingface_token (str): Hugging Face API token for authenticated access.
            dataset_name (str): Name of the dataset to be downloaded from Hugging Face.
            language_abbr (str): Language abbreviation for the dataset.
            model_id (str): Model ID for the model to be used in training.
            processing_task (str): The processing task to be performed (e.g., "transcribe").
        """
        self.huggingface_read_token = huggingface_read_token
        self.dataset_name = dataset_name
        self.language_abbr = language_abbr
        self.model_id = model_id
        self.processing_task = processing_task
        self.use_peft = use_peft
        self.model_prep = WhisperModelPrep(
            self.model_id,
            self.processing_task,
            self.use_peft,
        )
        self.data_loader = Dataset(
            self.huggingface_read_token, self.dataset_name, self.language_abbr
        )

    def prepare_model(
        self,
    ) -> Tuple[
        WhisperFeatureExtractor,
        WhisperTokenizer,
        WhisperProcessor,
        WhisperForConditionalGeneration,
    ]:
        """Initializes and sets up the necessary components for model training, including the tokenizer,
        feature extractor, feature processor, and the model itself.

        Returns:
        tuple: A tuple containing four elements in the order of tokenizer, feature extractor,
               feature processor, and the model. Each element is an initialized instance of its
               respective class, ready for use in the training pipeline.
        """

        self.tokenizer = self.model_prep.initialize_tokenizer()
        self.feature_extractor = self.model_prep.initialize_feature_extractor()
        self.feature_processor = self.model_prep.initialize_processor()
        self.model = self.model_prep.initialize_model()
        return (
            self.tokenizer,
            self.feature_extractor,
            self.feature_processor,
            self.model,
        )

    def load_dataset(self, feature_extractor: WhisperFeatureExtractor, tokenizer: WhisperTokenizer, processor: WhisperProcessor) -> DatasetDict:
        """
        Retrieves and preprocesses the specified dataset for model training and evaluation.

        Parameters:
        feature_extractor (PreTrainedFeatureExtractor): The feature extractor instance to be used for
                                                        audio data preprocessing.
        tokenizer (PreTrainedTokenizer): The tokenizer instance for processing textual data associated
                                         with the audio samples.

        Returns:
            DatasetDict: A dictionary containing the preprocessed 'train' and 'test' splits of the dataset,
                        ready for use in model training and evaluation. Each split has been cleaned and
                        processed to include only the necessary features for model input.
        """

        dataset = self.data_loader.load_dataset()
        print(f"Training dataset size: {self.data_loader.count_examples(dataset['train'])}")
        print(f"Test dataset size: {self.data_loader.count_examples(dataset['test'])}")
        processor = AudioDataProcessor(dataset, feature_extractor, tokenizer, processor)
        dataset['train']= dataset['train'].map(processor.resampled_dataset, remove_columns=list(next(iter(dataset['train'])).keys()))
        dataset['test']= dataset['test'].map(processor.resampled_dataset)
        return dataset

