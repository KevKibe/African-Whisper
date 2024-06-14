from transformers import WhisperForConditionalGeneration
from peft import PeftModel, PeftConfig


def merge_lora_weights(lora_model, output_dir, huggingface_write_token):
    peft_config = PeftConfig.from_pretrained(lora_model)
    base_model = WhisperForConditionalGeneration.from_pretrained(peft_config.base_model_name_or_path)
    model = PeftModel.from_pretrained(base_model, lora_model)
    model = model.merge_and_unload()
    model.train(False)
    model.push_to_hub(repo_id = output_dir, token = huggingface_write_token)
