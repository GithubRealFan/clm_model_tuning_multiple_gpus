import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import Trainer, TrainingArguments

# Load the pre-trained GPT-2 model and tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Example of using a custom dataset for fine-tuning
# Replace this with your own data preparation and tokenization
train_texts = ["Who are you?"]
val_texts = ["I am Ronaldo."]

train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)

train_dataset = torch.utils.data.TensorDataset(
    torch.tensor(train_encodings.input_ids),
    torch.tensor(train_encodings.attention_mask)
)

val_dataset = torch.utils.data.TensorDataset(
    torch.tensor(val_encodings.input_ids),
    torch.tensor(val_encodings.attention_mask)
)

# Training arguments
training_args = TrainingArguments(
    output_dir="./gpt2-fine-tuned",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    save_steps=500,
    save_total_limit=2,
    logging_steps=100,
    logging_dir="./logs",
    evaluation_strategy="steps",
    eval_steps=500,
    load_best_model_at_end=True,
)

# Instantiate the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Train the model
trainer.train()

# Save the fine-tuned model
trainer.save_model("./gpt2-fine-tuned")

# You can now use the fine-tuned model for text generation or other tasks.
