import torch
import deepspeed
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "tiiuae/falcon-40b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

deepspeed_config_file = "conf/deepspeed_config.json"

# Initialize the model and tokenizer using DeepSpeed
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
model = model.to('cuda')  # Move the model to the first GPU (rank 0)

# Initialize DeepSpeed with the proper world_size
model, _, _, _ = deepspeed.initialize(model=model, config_params=deepspeed_config_file, world_size=8)

# Wrap the model with DistributedDataParallel
model = torch.nn.parallel.DistributedDataParallel(
    model,
    device_ids=[torch.cuda.current_device()],
    output_device=torch.cuda.current_device(),
)

# Create the pipeline
pipeline = transformers.pipeline(
    "text-generation",
    model=model.module,  # Use the module attribute when wrapping with DistributedDataParallel
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
)

# Input text for generation
input_text = (
    "Girafatron is obsessed with giraffes, the most glorious animal on the face of this Earth. "
    "Giraftron believes all other animals are irrelevant when compared to the glorious majesty of the giraffe.\n"
    "Daniel: Hello, Girafatron!\nGirafatron:"
)

# Generate text using DeepSpeed-powered pipeline
sequences = pipeline(
    input_text,
    max_length=200,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
)

for seq in sequences:
    print(f"Result: {seq['generated_text']}")
