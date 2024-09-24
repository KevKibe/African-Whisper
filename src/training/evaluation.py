import evaluate
import warnings
from typing import List, Dict
import numpy as np
warnings.filterwarnings("ignore")

class MetricComputer:
    """
    A class to compute metrics for model predictions.

    This class encapsulates the functionality of loading a specific metric from the
    `evaluate` library and computing it using predictions and labels processed by a tokenizer.

    """

    def __init__(self, metric_name: str, tokenizer):
        """
        Initializes the MetricComputer with a specified metric and tokenizer.

        Args:
            metric_name (str): The name of the metric to load (e.g., "wer" for word error rate).
            tokenizer: The tokenizer to use for decoding prediction and label IDs.
        """
        self.metric = evaluate.load(metric_name)
        self.tokenizer = tokenizer

    def compute_metrics(self, pred) -> dict:
        """
        Computes the metric for a set of predictions and labels.

        The function processes the predictions and labels, replacing any padding token IDs with -100,
        and decodes them using the provided tokenizer before computing the metric.

        Args:
            pred: An object that contains `predictions` and `label_ids`, which are used to compute the metric.
                  The `predictions` and `label_ids` should be arrays of IDs.

        Returns:
            dict: A dictionary containing the computed metric value.
        """
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = self.tokenizer.pad_token_id
        pred_str = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = self.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        wer = 100 * self.metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}

###############################

def log_metric(
    accelerator,
    metrics: Dict,
    train_time: float,
    prefix: str = "eval",
):
    """Helper function to log all evaluation metrics with the correct prefixes and styling."""
    log_metrics = {}
    for k, v in metrics.items():
        log_metrics[f"{prefix}/{k}"] = v
    log_metrics[f"{prefix}/time"] = train_time
    accelerator.log(log_metrics)


def log_pred(
    accelerator,
    pred_str: List[str],
    label_str: List[str],
    norm_pred_str: List[str],
    norm_label_str: List[str],
    prefix: str = "eval",
    num_lines: int = 200000,
):
    """Helper function to log target/predicted transcriptions to weights and biases (wandb)."""
    if accelerator.is_main_process:
        wandb_tracker = accelerator.get_tracker("wandb")
        # pretty name for split
        prefix = prefix.replace("/", "-")

        # convert str data to a wandb compatible format
        str_data = [[label_str[i], pred_str[i], norm_label_str[i], norm_pred_str[i]] for i in range(len(pred_str))]
        # log as a table with the appropriate headers
        wandb_tracker.log_table(
            table_name=f"{prefix}/all_predictions",
            columns=["Target", "Pred", "Norm Target", "Norm Pred"],
            data=str_data[:num_lines],
        )

        # log incorrect normalised predictions
        str_data = np.asarray(str_data)
        str_data_incorrect = str_data[str_data[:, -2] != str_data[:, -1]]
        # log as a table with the appropriate headers
        wandb_tracker.log_table(
            table_name=f"{prefix}/incorrect_predictions",
            columns=["Target", "Pred", "Norm Target", "Norm Pred"],
            data=str_data_incorrect[:num_lines],
        )


def compute_metrics(
        preds,
        labels,
        file_ids,
        tokenizer,
        normalizer,
        return_timestamps=False
):
    # replace padded labels by the padding token
    for idx in range(len(labels)):
        labels[idx][labels[idx] == -100] = tokenizer.pad_token_id

    pred_str = tokenizer.batch_decode(preds, skip_special_tokens=False, decode_with_timestamps=return_timestamps)
    # we do not want to group tokens when computing the metrics
    label_str = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # normalize everything and re-compute the WER
    norm_pred_str = [normalizer(pred) for pred in pred_str]
    norm_label_str = [normalizer(label) for label in label_str]
    # for logging, we need the pred/labels to match the norm_pred/norm_labels, so discard any filtered samples here
    pred_str = [pred_str[i] for i in range(len(norm_pred_str)) if len(norm_label_str[i]) > 0]
    label_str = [label_str[i] for i in range(len(norm_label_str)) if len(norm_label_str[i]) > 0]
    file_ids = [file_ids[i] for i in range(len(file_ids)) if len(norm_label_str[i]) > 0]
    # filtering step to only evaluate the samples that correspond to non-zero normalized references:
    norm_pred_str = [norm_pred_str[i] for i in range(len(norm_pred_str)) if len(norm_label_str[i]) > 0]
    norm_label_str = [norm_label_str[i] for i in range(len(norm_label_str)) if len(norm_label_str[i]) > 0]
    metric = evaluate.load("wer")
    wer = 100 * metric.compute(predictions=norm_pred_str, references=norm_label_str)

    return {"wer": wer}, pred_str, label_str, norm_pred_str, norm_label_str, file_ids