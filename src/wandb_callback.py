from transformers import TrainerCallback
from transformers.integrations import WandbCallback
import holoviews as hv
# from bokeh.resources import INLINE      
hv.extension("bokeh", logo=False)
import panel as pn
import numpy as np
from io import StringIO
import wandb
from pathlib import Path
import pandas as pd
import jiwer
import tempfile
import numbers


def record_to_html(sample_record):
    audio_array = np.array(sample_record["audio"]["array"])
    audio_sr = sample_record["audio"]["sampling_rate"]
    audio_data = sample_record['audio']
    audio_duration = len(audio_data) / 16000
    audio_spectrogram = np.array(sample_record["spectrogram"])

    bounds = (0,0, audio_duration, audio_spectrogram.max())

    waveform_int = np.int16(audio_array * 32767)

    
    
    hv_audio = pn.pane.Audio(waveform_int, sample_rate=audio_sr, name='Audio', throttle=500)
    
    slider = pn.widgets.FloatSlider(end=audio_duration, visible=False, step=0.001)
    line_audio = hv.VLine(0).opts(color='black')
    line_spec = hv.VLine(0).opts(color='red')
    
    
    slider.jslink(hv_audio, value='time', bidirectional=True)
    slider.jslink(line_audio, value='glyph.location')
    slider.jslink(line_spec, value='glyph.location')
    
    time = np.linspace(0, audio_duration, num=len(audio_array))
    line_plot_hv = hv.Curve(
        (time, audio_array), ["Time (s)", "amplitude"]).opts(
        width=500, height=150, axiswise=True) * line_audio
    
    hv_spec_gram = hv.Image(
        audio_spectrogram, bounds=(bounds), kdims=["Time (s)", "Frequency (hz)"]).opts(
        width=500, height=150, labelled=[], axiswise=True, color_levels=512)* line_spec
    
    
    combined = pn.Row(hv_audio, hv_spec_gram, line_plot_hv, slider)
    audio_html = StringIO()
    combined.save(audio_html)
    return audio_html


def dataset_to_records(dataset):
    records = []
    for item in dataset:
        record = {}
        audio_data = item['audio'] 
        audio_duration = len(audio_data) / 16000 
        record["audio_with_spec"] = wandb.Html(record_to_html(item))
        record["sentence"] = item["sentence"]
        record["length"] = audio_duration
        records.append(record)
    records = pd.DataFrame(records)
    return records


def decode_predictions( predictions, tokenizer):
    pred_ids = predictions.predictions
    print(type(tokenizer))
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True, )
    return pred_str

def compute_measures(predictions, labels,):
    measures = [jiwer.compute_measures(ls, ps) for ps, ls in zip(predictions, labels)]
    measures_df = pd.DataFrame(measures)[["wer", "hits", "substitutions", "deletions", "insertions"]]
    return measures_df


class WandbProgressResultsCallback(WandbCallback):
    def __init__(self, trainer, sample_dataset, tokenizer): 
        super().__init__()
        self.trainer = trainer
        self.sample_dataset = sample_dataset
        self.records_df = dataset_to_records(sample_dataset)
        self.tokenizer = tokenizer
        
    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        super().on_log(args, state, control, model, logs)
        predictions = self.trainer.predict(self.sample_dataset)
        predictions = decode_predictions(predictions, self.tokenizer)
        measures_df = compute_measures(predictions, self.records_df["sentence"].tolist())
        records_df = pd.concat([self.records_df, measures_df], axis=1)
        records_df["prediction"] = predictions
        records_df["step"] = state.global_step
        records_table = self._wandb.Table(dataframe=records_df)
        self._wandb.log({"sample_predictions": records_table})
        
    def on_save(self, args, state, control, model=None, tokenizer=None, **kwargs):
        if self._wandb is None:
            return
        if self._log_model and self._initialized and state.is_world_process_zero:
            with tempfile.TemporaryDirectory() as temp_dir:
                self.trainer.save_model(temp_dir)
                metadata = (
                    {
                        k: v
                        for k, v in dict(self._wandb.summary).items()
                        if isinstance(v, numbers.Number) and not k.startswith("_")
                    }
                    if not args.load_best_model_at_end
                    else {
                        f"eval/{args.metric_for_best_model}": state.best_metric,
                        "train/total_floss": state.total_flos,
                    }
                )
                artifact = self._wandb.Artifact(
                    name=f"model-{self._wandb.run.id}",
                    type="model", metadata=metadata)
                for f in Path(temp_dir).glob("*"):
                    if f.is_file():
                        with artifact.new_file(f.name, mode="wb") as fa:
                            fa.write(f.read_bytes())
                self._wandb.run.log_artifact(artifact)