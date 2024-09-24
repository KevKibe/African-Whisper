from transformers import WhisperForConditionalGeneration
from peft import PeftModel, PeftConfig
import argparse


class Merger:
    def merge_lora_weights(hf_model_id, huggingface_token):
        """
        Merge LoRA weights with a pre-trained Whisper model and upload the merged model to the Hugging Face Hub.

        Args:
            hf_model_id (str): The Hugging Face model ID for the LoRA configuration.
            huggingface_token (str): The Hugging Face write token for authentication.

        Returns:
            PeftModel: The merged model.
        """
        peft_config = PeftConfig.from_pretrained(hf_model_id)
        base_model = WhisperForConditionalGeneration.from_pretrained(peft_config.base_model_name_or_path)
        model = PeftModel.from_pretrained(base_model, hf_model_id)
        model = model.merge_and_unload()
        model.train(False)
        model.push_to_hub(repo_id=hf_model_id, token=huggingface_token)
        print(f"{hf_model_id} LoRA weights merged")


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA weights with a pre-trained Whisper model and upload to the Hugging Face Hub.")
    parser.add_argument("--hf_model_id", type=str, help="The Hugging Face model ID for the LoRA configuration.")
    parser.add_argument("--huggingface_token", type=str, help="The Hugging Face write token for authentication.")
    args = parser.parse_args()

    Merger.merge_lora_weights(args.hf_model_id, args.huggingface_token)

if __name__ == "__main__":
    main()