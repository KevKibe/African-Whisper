from transformers import WhisperForConditionalGeneration
from peft import prepare_model_for_int8_training
from peft import LoraConfig, PeftModel, LoraModel, LoraConfig, get_peft_model


def make_inputs_require_grad(module, input, output):
    output.requires_grad_(True)
def model_prep():

    # model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small", load_in_8bit=True, device_map="auto")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []


    # model = prepare_model_for_int8_training(model, "proj_out")
    # model.model.encoder.conv1.register_forward_hook(make_inputs_require_grad)
    # config = LoraConfig(r=32, lora_alpha=64, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none")
 
    # model = get_peft_model(model, config)
    # model.print_trainable_parameters()
    return model



print(model_prep())

