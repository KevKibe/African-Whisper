from transformers import WhisperForConditionalGeneration
from peft import prepare_model_for_int8_training
from peft import LoraConfig, PeftModel, LoraModel, LoraConfig, get_peft_model



model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small", load_in_8bit=True, device_map="auto")
model = prepare_model_for_int8_training(model, "proj_out")
def make_inputs_require_grad(module, input, output):
    output.requires_grad_(True)

model.model.encoder.conv1.register_forward_hook(make_inputs_require_grad)
config = LoraConfig(r=32, lora_alpha=64, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none")

model = get_peft_model(model, config)
model.print_trainable_parameters()


