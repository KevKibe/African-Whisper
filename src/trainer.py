from transformers import Seq2SeqTrainingArguments
from load_data import Dataset
from transformers import Seq2SeqTrainer
from model_prep import ModelPrep
from collator import DataCollatorSpeechSeq2SeqWithPadding
from evaluation import MetricComputer
import evaluate

language_abbr = "sw"
dataset_name = "mozilla-foundation/common_voice_16_1"
huggingface_token = "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"
data_loader = Dataset(huggingface_token, dataset_name, language_abbr)
dataset = data_loader.load_dataset()

model_id = "openai/whisper-small"
processing_task = "transcribe"
model_prep = ModelPrep(dataset, model_id, language_abbr, processing_task)
tokenizer = model_prep.initialize_tokenizer()
feature_extractor = model_prep.initialize_feature_extractor()
feature_processor = model_prep.initialize_processor()
model = model_prep.initialize_model()

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=feature_processor)


metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-small-sw",  
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-5,
    warmup_steps=50,
    max_steps=100,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=1000,
    eval_steps=1000,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=True,
)


# trainer = Seq2SeqTrainer(
#     args=training_args,
#     model=model,
#     train_dataset=dataset["train"],
#     eval_dataset=dataset["test"],
#     data_collator=data_collator,
#     compute_metrics=compute_metrics,
#     tokenizer=feature_processor.feature_extractor,
# )
trainer = Seq2SeqTrainingArguments(
    output_dir="./whisper-small-sw",  
    per_device_train_batch_size=16,  
    gradient_accumulation_steps=1,  
    learning_rate=1e-5,
    warmup_steps=50,
    max_steps=50,  
    gradient_checkpointing=True,
    fp16=True,  
    evaluation_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=25,  
    eval_steps=25,  
    logging_steps=10,  
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False, 
)


trainer.train()
