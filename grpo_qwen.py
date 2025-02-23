import os
import re
import torch
from datasets import load_dataset, Dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl.trainer import GRPOConfig, GRPOTrainer
import wandb




R1_STYLE_SYSTEM_PROMPT = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
The assistant first thinks about the reasoning process in the mind and then provides the user
with the answer. The reasoning process and answer are enclosed within <reasoning> </reasoning> and
<answer> </answer> tags, respectively, i.e., <reasoning> reasoning process here </reasoning>
<answer> answer here </answer>."""

TASK_SPECIFIC_INSTRUCTIONS = "The answer must be a single integer."


def preprocess_dataset(dataset_name, split = "train", chunk_size = 1000) -> Dataset:
    dataset = load_from_disk(dataset_name)[split]

    def extract_hash_answer(text: str) -> str | None:
        try:
            return text.split("####")[1].strip()
        except IndexError:
            return None

    def process_batch(batch):
        prompts = [[
            {'role': 'system', 'content': R1_STYLE_SYSTEM_PROMPT + "\n" + TASK_SPECIFIC_INSTRUCTIONS},
            {'role': 'user', 'content': "What is 2+2?"},
            {'role': 'assistant', 'content': "<reasoning>To calculate 2+2, we simply add the numbers together: 2 + 2 = 4.</reasoning>\n<answer>4</answer>"},
            {'role': 'user', 'content': q.strip()}
        ] for q in batch['question']]

        return {
            'prompt': prompts,
            'answer': [extract_hash_answer(a) for a in batch['answer']]
        }

    return dataset.map(process_batch, batched = True, batch_size = chunk_size)



def extract_xml_answer(text: str) -> str:
    try:
        answer = text.split("<answer>")[-1].split("</answer>")[0].strip()
        return answer
    except IndexError:
        return ""

def format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has the correct format."""
    pattern = r"^<reasoning>(?:(?!</reasoning>).)*</reasoning>\n<answer>(?:(?!</answer>).)*</answer>$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [bool(re.match(pattern, r)) for r in responses]
    return [1.0 if match else 0.0 for match in matches]


def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    """Reward function that checks if the answer is correct."""
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print(f"\n\n===============================================================\n"  
            f"User Question:\n{prompts[0][-1]['content']}"
            f"\n\nCorrect Answer:\n{answer[0]}\n"
            f"\n---------------------------------------------------------------\n"
            f"\n\n1st/{len(completions)} generated responses:\n{responses[0]}"
            f"\n\nExtracted: {extracted_responses[0]}"
            f"\n\nCorrectness of all {len(completions)} responses: " + ''.join('Y' if r == a else 'N' for r, a in zip(extracted_responses, answer)))

    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]




def main():
    dataset_name = './dataset/gsm8k'
    dataset = preprocess_dataset(dataset_name, chunk_size=500)

    model_name = "./../../../scratch/gpfs/my0049/Qwen2.5-1.5B-Instruct"

    output_dir = f"./../../../scratch/gpfs/my0049/{model_name.split('/')[-1]}-GRPO"
    run_name = f"{model_name.split('/')[-1]}-{dataset_name.split('/')[-1]}"

    # Set memory-related environment variables
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

    training_args = GRPOConfig(
        learning_rate = 1e-5,
        beta = 0.005, # divergence coefficient – how much the policy is allowed to deviate from the reference model. higher value – more conservative updates. Default is 0.04
        optim = "adamw_8bit",
        adam_beta1 = 0.9,
        adam_beta2 = 0.99,
        weight_decay = 0.1,
        warmup_ratio = 0.1,
        lr_scheduler_type = 'cosine',
        logging_steps = 1,
        bf16 = True,
        per_device_train_batch_size = 4,
        num_generations = 4,  # group size
        gradient_accumulation_steps = 4,
        max_prompt_length = 256,
        max_completion_length = 512,
        num_train_epochs = 1,
        save_steps = 1000,
        max_grad_norm = 0.1,
        report_to = "wandb",
        output_dir = output_dir,
        run_name = run_name,
        log_on_each_node = False,
    )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype = torch.bfloat16,
        # attn_implementation = "flash_attention_2",
        device_map = "auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        model_max_length = training_args.max_completion_length,
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Initialize trainer
    trainer = GRPOTrainer(
        model = model,
        processing_class = tokenizer,
        reward_funcs = [
            format_reward_func,
            correctness_reward_func
        ],
        args = training_args,
        train_dataset = dataset,
    )

    wandb.init(project="deepseek_r1_zero_grpo", name=run_name, mode="offline")  # name specifies job name
    trainer.train()
    trainer.save_model(training_args.output_dir)



if __name__ == "__main__":
    main()