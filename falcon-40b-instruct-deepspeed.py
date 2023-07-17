import torch
import deepspeed
from transformers import AutoTokenizer, AutoModelForCausalLM

# Model and tokenizer
model_name = "tiiuae/falcon-40b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# DeepSpeed configuration
deepspeed_config = {
    "train_batch_size": 18,
    "gradient_accumulation_steps": 1,
    "optimizer": {
        "type": "Adam",
        "params": {
            "lr": 5e-6,
            "weight_decay": 0.0
        }
    },
    "fp16": {
        "enabled": True,
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "hysteresis": 2,
        "min_loss_scale": 1
    },
    "steps_per_print": 80,
    "wall_clock_breakdown": True
}

# Initialize DeepSpeed
model, _, _, _ = deepspeed.initialize(model=model, config_params=deepspeed_config)

# DeepSpeed pipeline
pipeline = torch.distributed.DistributedDataParallel(model, device_ids=[torch.cuda.current_device()])

# Input text
input_text = "Girafatron is obsessed with giraffes, the most glorious animal on the face of this Earth. Giraftron believes all other animals are irrelevant when compared to the glorious majesty of the giraffe.\nDaniel: Hello, Girafatron!\nGirafatron:"

# Tokenize input text
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(pipeline.device)

# Generate output text
output_ids = pipeline.generate(
    input_ids,
    max_length=200,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id
)

# Decode and print the generated text
output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print("Result:", output_text)
