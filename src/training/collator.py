import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import warnings

warnings.filterwarnings("ignore")

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    A data collator that dynamically pads a batch of inputs for speech-to-text sequence-to-sequence models.

    Attributes:
        processor (Any): The processor combining feature extraction and tokenization,
                         used for preparing model inputs and labels.
    """

    processor: Any

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        """
        Processes and pads a batch of input features and labels for training or evaluation.

        Args:
            features (List[Dict[str, Union[List[int], torch.Tensor]]]): A list of feature dictionaries,
                each containing 'input_features' for model inputs and 'labels' for expected outputs.

        Returns:
            Dict[str, torch.Tensor]: A dictionary of padded tensors including 'input_features' for the model inputs
                                     and 'labels' for the model outputs, with padding applied according to the processor's
                                     configurations.
        """
        input_features = [
            {"input_features": feature["input_features"]} for feature in features
        ]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch
