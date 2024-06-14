from transformers import WhisperForConditionalGeneration
from peft import PeftModel, PeftConfig


def merge_lora_weights(hf_model_id, output_dir, huggingface_write_token):
    # Load the model configuration from Hugging Face
    peft_config = PeftConfig.from_pretrained(hf_model_id)

    # Load the base model and the LoRA model
    base_model = WhisperForConditionalGeneration.from_pretrained(peft_config.base_model_name_or_path)
    model = PeftModel.from_pretrained(base_model, hf_model_id)

    # Merge the weights and unload the LoRA model
    model = model.merge_and_unload()
    model.train(False)

    # Push the model to the Hugging Face Model Hub
    model.push_to_hub(repo_id=output_dir, token=huggingface_write_token)

    return model
