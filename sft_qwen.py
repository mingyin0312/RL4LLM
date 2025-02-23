import os
import re
import torch
from datasets import load_from_disk, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig
import wandb

# System prompt and task instructions
R1_STYLE_SYSTEM_PROMPT = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
The assistant first thinks about the reasoning process in the mind and then provides the user
with the answer. The reasoning process and answer are enclosed within <reasoning> </reasoning> and
<answer> </answer> tags, respectively, i.e., <reasoning> reasoning process here </reasoning>
<answer> answer here </answer>."""
TASK_SPECIFIC_INSTRUCTIONS = "The answer must be a single integer."

def convert_chat_to_text(chat: list) -> str:
    """
    Convert a list of chat messages (each a dict with 'role' and 'content')
    into a single formatted string.
    """
    lines = []
    for message in chat:
        role = message["role"].capitalize()
        content = message["content"]
        lines.append(f"{role}: {content}")
    return "\n".join(lines)

def preprocess_dataset(dataset_name, split="train", chunk_size=1000) -> Dataset:
    """
    Load the dataset from disk and process each batch to generate chat-style prompts.
    The resulting dataset will have a "text" field (a string) and an "answer" field.
    """
    dataset = load_from_disk(dataset_name)[split]

    def extract_hash_answer(text: str) -> str | None:
        try:
            return text.split("####")[1].strip()
        except IndexError:
            return None

    def process_batch(batch):
        chats = [
            convert_chat_to_text([
                {'role': 'system', 'content': R1_STYLE_SYSTEM_PROMPT + "\n" + TASK_SPECIFIC_INSTRUCTIONS},
                {'role': 'user', 'content': "What is 2+2?"},
                {'role': 'assistant', 'content': "<reasoning>To calculate 2+2, we simply add the numbers together: 2 + 2 = 4.</reasoning>\n<answer>4</answer>"},
                {'role': 'user', 'content': q.strip()}
            ])
            for q in batch['question']
        ]
        return {
            'text': chats,  # Renamed field for SFTTrainer tokenization
            'answer': [extract_hash_answer(a) for a in batch['answer']]
        }

    return dataset.map(process_batch, batched=True, batch_size=chunk_size)

def main():
    # Load and preprocess the dataset
    dataset_name = './dataset/gsm8k'
    dataset = preprocess_dataset(dataset_name, chunk_size=500)

    # Define model and output paths
    model_name = "./../../../scratch/gpfs/my0049/Qwen2.5-1.5B-Instruct"
    output_dir = f"./../../../scratch/gpfs/my0049/{model_name.split('/')[-1]}-SFT"
    run_name = f"{model_name.split('/')[-1]}-{dataset_name.split('/')[-1]}"

    # Set memory-related environment variable
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

    # Create SFT training configuration
    training_args = SFTConfig(
        learning_rate=1e-6,
        logging_steps=1,
        bf16=True,
        per_device_train_batch_size=4,
        num_train_epochs=2,
        save_steps=10**6, 
        max_grad_norm=0.1,
        report_to="wandb",
        output_dir=output_dir,
        run_name=run_name,
        log_on_each_node=False,
    )

    # Load the model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Load the tokenizer and set pad token
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        model_max_length=512,
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Initialize the SFT trainer using the tokenizer as the processing class
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )

    # Initialize wandb in offline mode for experiment tracking
    wandb.init(project="deepseek_r1_zero_sft", name=run_name, mode="offline")
    trainer.train()
    trainer.save_model(training_args.output_dir)

if __name__ == "__main__":
    main()