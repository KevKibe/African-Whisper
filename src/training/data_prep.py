from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
)
from transformers.models.whisper.english_normalizer import BasicTextNormalizer, EnglishTextNormalizer
from huggingface_hub import create_repo, get_full_repo_name, Repository
from pathlib import Path
import numpy as np
import datasets
import os
from datasets import (
    DatasetDict,
    IterableDatasetDict,
)
from soundfile import LibsndfileError
from datasets.arrow_dataset import table_iter
from accelerate.logging import get_logger
from .load_data import Dataset
from .whisper_model_prep import WhisperModelPrep
from .audio_data_processor import AudioDataProcessor
from typing import Tuple, List
import warnings
warnings.filterwarnings("ignore")

logger = get_logger(__name__)



class DataPrep:
    """

    A class to encapsulate preparing the speech dataset for model training, including dataset loading, cleaning, and preprocessing tasks like feature extraction and tokenization. .

    """

    def __init__(
        self,
        huggingface_token: str,
        dataset_name: str,
        language_abbr: List[str],
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
        self.huggingface_token = huggingface_token
        self.dataset_name = dataset_name
        self.language_abbr = language_abbr
        self.model_id = model_id
        self.processing_task = processing_task
        self.use_peft = use_peft
        self.model_prep = WhisperModelPrep(
            language = self.language_abbr,
            model_id=self.model_id,
            processing_task=self.processing_task,
            use_peft=self.use_peft,
        )
        self.data_loader = Dataset(
            self.huggingface_token, self.dataset_name, self.language_abbr
        )

    def prepare_model(
        self,
    ) -> Tuple[
        WhisperTokenizer,
        WhisperFeatureExtractor,
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

    def load_dataset(
        self,  
        feature_extractor: WhisperFeatureExtractor, 
        tokenizer: WhisperTokenizer, 
        processor: WhisperProcessor,
        streaming: bool = True,
        train_num_samples: int = None,
        test_num_samples: int = None) -> dict:
        """
        Retrieves and preprocesses the specified dataset for model training and evaluation.

        Parameters:
        feature_extractor (PreTrainedFeatureExtractor):
            The feature extractor instance to be used for audio data preprocessing.
        tokenizer (PreTrainedTokenizer):
            The tokenizer instance for processing textual data associated with the audio samples.
        processor (WhisperProcessor):
            The processor instance for processing the audio samples.
        streaming (bool):
            If True, the datasets will be streamed, allowing for loading large datasets without requiring them to fit into memory.
            If False, the entire dataset will be downloaded before processing.
        train_num_samples (int, optional):
            The maximum number of training samples to load from each dataset.
            For example, if train_num_samples = 100, then 100 samples will be loaded from each dataset's training split.
            If None, the entire training split will be loaded.
        test_num_samples (int, optional):
            The maximum number of test samples to load from each dataset.
            For example, if test_num_samples = 100, then 100 samples will be loaded from each dataset's test split.
            If None, the entire test split will be loaded.

        Returns:
            dict: A dictionary containing the preprocessed 'train' and 'test' splits of the dataset,
                        ready for use in model training and evaluation. Each split has been cleaned and
                        processed to include only the necessary features for model input.
        """

        dataset = self.data_loader.load_dataset(
            streaming,
            train_num_samples = train_num_samples,
            test_num_samples = test_num_samples
        )
        print(dataset)
        # train_count, test_count = self.data_loader.count_examples(dataset)
        # print(f"Training dataset size: {train_count}")
        # print(f"Test dataset size: {test_count}")
        processor = AudioDataProcessor(dataset, feature_extractor, tokenizer, processor)
        dataset['train']= dataset['train'].map(processor.resampled_dataset, remove_columns=list(next(iter(dataset['train'])).keys()))
        dataset['test']= dataset['test'].map(processor.resampled_dataset)
        return dataset


##########################

def preprocess_datasets(
    model,
    feature_extractor,
    tokenizer,
    is_multilingual,
    language,
    data_splits,
    raw_datasets,
    accelerator,
    token,
    dataset_name=None,
    max_duration_in_seconds=30.0,
    max_label_length=256,
    audio_column_name="audio",
    preprocessing_batch_size=500,
    preprocessing_num_workers=None,
    dataloader_num_workers=0,
    text_column_name="sentence",
    id_column_name="id",
    speaker_id_column_name=None,
    max_samples_per_split=None,
    streaming=False,
    concatenate_audio=True,
    preprocessing_only=False,
    output_dir=None,
    push_to_hub=True,
    hub_model_id=None
):
    max_input_length = int(max_duration_in_seconds * feature_extractor.sampling_rate)
    max_label_length = (
        max_label_length if max_label_length is not None else model.config.max_length
    )
    audio_column_name = audio_column_name
    sampling_rate = feature_extractor.sampling_rate

    preprocessing_batch_size = preprocessing_batch_size
    num_workers = preprocessing_num_workers
    dataloader_num_workers = dataloader_num_workers

    text_column_name = text_column_name
    model_input_name = feature_extractor.model_input_names[0]
    id_column_name = id_column_name
    speaker_id_column_name = speaker_id_column_name
    normalizer = (
        BasicTextNormalizer()
        if language is not None
        else EnglishTextNormalizer(tokenizer.english_spelling_normalizer)
    )

    timestamp_position = 3 if is_multilingual else 1
    decoder_prev_token_id = tokenizer.convert_tokens_to_ids("<|startofprev|>")
    decoder_eot_token_id = tokenizer.eos_token_id

    if max_samples_per_split is not None:
        for split in data_splits:
            raw_datasets[split] = (
                raw_datasets[split].take(max_samples_per_split)
                if streaming
                else raw_datasets[split].select(range(max_samples_per_split))
            )

    if speaker_id_column_name is not None:
        raw_datasets = raw_datasets.sort(speaker_id_column_name)


    def concatenate_dataset(batch):
        audio_arrays, texts, speaker_ids = [], [], []

        # skip corrupted samples
        for row in table_iter(batch.pa_table, batch_size=1):
            row = batch.formatter.format_row(row)
            try:
                sample_audio = row[audio_column_name]['array']
                sample_text = row[text_column_name]
                sample_speaker_id = row[speaker_id_column_name] if speaker_id_column_name else None
            except LibsndfileError:
                logger.warning(f"{row[id_column_name]} is corrupted! Skipping sample.")
                continue
            audio_arrays.append(sample_audio)
            texts.append(sample_text)
            speaker_ids.append(sample_speaker_id)

        # initialize concatenations
        concat_audio = [audio_arrays[0]]
        concat_text = [texts[0]]
        concat_speaker_id = [speaker_ids[0]]
        condition_on_prev = [0]

        for audio_array, text, speaker_id in zip(audio_arrays[1:], texts[1:], speaker_ids[1:]):
            is_same_speaker = speaker_id == concat_speaker_id[-1]
            is_concatenable = len(audio_array) + len(concat_audio[-1]) <= max_input_length
            if is_same_speaker and is_concatenable:
                # inplace concatenation
                concat_audio[-1] = np.append(concat_audio[-1], audio_array)
                concat_text[-1] = concat_text[-1] + " " + text
            else:
                concat_audio.append(audio_array)
                concat_text.append(text)
                concat_speaker_id.append(speaker_id)
                condition_on_prev.append(1 if is_same_speaker else 0)

        batch[audio_column_name] = [{"array": array, "sampling_rate": sampling_rate} for array in concat_audio]
        batch[text_column_name] = concat_text
        batch[id_column_name] = concat_speaker_id
        batch["condition_on_prev"] = condition_on_prev

        return batch

    raw_datasets_features = list(next(iter(raw_datasets.values())).features.keys())
    if concatenate_audio and not streaming:
        with accelerator.main_process_first():
            raw_datasets = raw_datasets.map(
                concatenate_dataset,
                batched=True,
                batch_size=preprocessing_batch_size,
                num_proc=num_workers,
                remove_columns=set(raw_datasets_features)
                               - {audio_column_name, text_column_name, id_column_name, "condition_on_prev"},
                desc="Concatenating dataset...",
            )

        raw_datasets = raw_datasets.cast_column(
            audio_column_name, datasets.features.Audio(sampling_rate=sampling_rate)
        )
        pretty_name = dataset_name.split("/")[-1]

        def postprocess_ids(speaker_ids, indices):
            speaker_ids_formatted = []
            for speaker, idx in zip(speaker_ids, indices):
                formatted_idx = f"{pretty_name}-{speaker}-{idx}" if speaker is not None else f"{pretty_name}-{idx}"
                speaker_ids_formatted.append(formatted_idx)
            return {id_column_name: speaker_ids_formatted}

        with accelerator.main_process_first():
            raw_datasets = raw_datasets.map(
                postprocess_ids,
                input_columns=[id_column_name],
                with_indices=True,
                desc="Setting sample idxs...",
                batched=True,
                batch_size=preprocessing_batch_size,
                num_proc=num_workers,
            )
    elif concatenate_audio and streaming:
        raise ValueError(
            "Streaming mode is not yet compatible with concatenating audios to `max_duration_in_seconds`."
            "Either set `--streaming=False` and download the audios locally, or open an issue on the Distil-Whisper repo to request this feature."
        )

    def prepare_dataset(batch):
        # process audio
        sample = batch[audio_column_name]
        inputs = feature_extractor(sample["array"], sampling_rate=sample["sampling_rate"])
        # process audio length
        batch[model_input_name] = inputs.get(model_input_name)[0]

        # process targets
        input_str = batch[text_column_name]
        batch["labels"] = tokenizer(input_str, max_length=max_label_length, truncation=True).input_ids
        return batch

    raw_datasets_features = list(next(iter(raw_datasets.values())).features.keys())
    file_ids_dataset = IterableDatasetDict() if streaming else DatasetDict()
    for split in raw_datasets:
        file_ids_dataset[split] = raw_datasets[split][id_column_name]
    if streaming:
        with accelerator.main_process_first():
            vectorized_datasets = raw_datasets.map(prepare_dataset, remove_columns=raw_datasets_features)
    else:
        with accelerator.main_process_first():
            vectorized_datasets = raw_datasets.map(
                prepare_dataset,
                remove_columns=raw_datasets_features,
                num_proc=num_workers,
                desc="preprocess dataset",
            )

    # for large datasets it is advised to run the preprocessing on a
    # single machine first with `args.preprocessing_only` since there will mostly likely
    # be a timeout when running the script in distributed mode.
    # In a second step `args.preprocessing_only` can then be set to `False` to load the
    # cached dataset
    if preprocessing_only:
        cache = {k: v.cache_files for k, v in vectorized_datasets.items()}
        logger.info(f"Data preprocessing finished. Files cached at {cache}.")
        return

    if streaming and dataloader_num_workers > 0:
        logger.warning(
            "Using multiple dataloader num workers with streaming mode will result in different shards of "
            "data being transcribed in parallel. This is not advised if you want to preserve the order of the "
            "audio-text data."
        )

    # Handle the repository creation
    repo_name = None
    output_dir = output_dir
    if accelerator.is_main_process:
        if push_to_hub:
            if hub_model_id is None:
                repo_name = get_full_repo_name(
                    Path(output_dir).absolute().name,
                    token=token,
                )
            else:
                repo_name = hub_model_id
            create_repo(repo_name, repo_type="dataset", exist_ok=True, token=token)
            Repository(
                output_dir,
                clone_from=repo_name,
                token=token,
                repo_type="dataset",
            )

            # Ensure large txt files can be pushed to the Hub with git-lfs
            with open(os.path.join(output_dir, ".gitattributes"), "r+") as f:
                git_lfs_extensions = f.read()
                if "*.csv" not in git_lfs_extensions:
                    f.write("*.csv filter=lfs diff=lfs merge=lfs -text")

        elif output_dir is not None:
            # this is where we'll save our transcriptions
            os.makedirs(output_dir, exist_ok=True)

    accelerator.wait_for_everyone()
    return vectorized_datasets, file_ids_dataset, decoder_eot_token_id, decoder_prev_token_id, timestamp_position, repo_name, normalizer

