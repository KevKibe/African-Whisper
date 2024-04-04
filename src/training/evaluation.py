import evaluate
import warnings

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
