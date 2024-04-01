from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
)
from .load_data import Dataset
from .whisper_model_prep import WhisperModelPrep
from .audio_data_processor import AudioDataProcessor
from datasets import DatasetDict, Audio, IterableDatasetDict
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
            self.language_abbr,
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

    def load_dataset(self, feature_extractor, tokenizer, processor) -> DatasetDict:
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
        dataset = IterableDatasetDict()
        # dataset= {}
        dataset["train"] = self.data_loader.load_streaming_dataset(split = "train")
        dataset["test"] = self.data_loader.load_streaming_dataset(split = "test")
        print(f"Training Dataset Size: {self.data_loader.count_examples(dataset['train'])}")
        print(f"Testing Dataset Size: {self.data_loader.count_examples(dataset['test'])}")
        dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
        processor = AudioDataProcessor(dataset, feature_extractor, tokenizer, processor)
        processed_datasets = dataset.map(processor.prepare_dataset, remove_columns=list(next(iter(dataset.values())).features)).with_format("torch")
        #TODO: Feaature in testing
        # processed_datasets = {}
        # for key, value in dataset.items():
        #     processed_datasets[key] = value.map(processor.prepare_dataset, remove_columns=list(value.features)).with_format("torch")

        # print(f"dataset : {processed_datasets}")
        # print(f"dataset : {processed_datasets['train']}")
        return processed_datasets

