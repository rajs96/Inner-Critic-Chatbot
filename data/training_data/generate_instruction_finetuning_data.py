from typing import List
import pandas as pd
import asyncio
from tqdm.asyncio import tqdm_asyncio
from config import INSTRUCTION_FINETUNING_GENERATION_CONFIG as CONFIG
from config.chains import INSTRUCTION_FINETUNING_GENERATION_CHAIN
from data.reddit.sample_reddit_data import two_phase_reddit_sampling


async def generate_instruction_output(input_message: str) -> str:
    return await INSTRUCTION_FINETUNING_GENERATION_CHAIN.arun(input_message)


async def generate_all_instruction_outputs(
    input_messages: List[str], request_batch_size: int = 65
) -> List[str]:
    instruction_outputs = []
    num_instances = len(input_messages)
    for i in tqdm_asyncio(
        range(0, num_instances, request_batch_size),
        desc="Generating instruction outputs for batch",
    ):
        tasks = []
        print(f"Started generating instruction outputs {i} to {i+request_batch_size}")
        for j in tqdm_asyncio(
            range(i, min(i + request_batch_size, num_instances)),
            desc="Generating instruction outputs for batch",
        ):
            tasks.append(generate_instruction_output(input_messages[j]))
            await asyncio.sleep(0.75)
        
        batch_instruction_outputs = await asyncio.gather(*tasks)
        print(f"Completed generating instances {i} to {i+request_batch_size}")
        instruction_outputs.extend(batch_instruction_outputs)
        await asyncio.sleep(5)

if __name__ == "__main__":
    filtered_reddit_df = pd.read_csv(CONFIG["reddit_data_filtered_path"])
    sampled_reddit_df = two_phase_reddit_sampling(
        reddit_df=filtered_reddit_df,
        negative_emotions_str=CONFIG["negative_emotions"],
        n_samples_phase_1=int(CONFIG["n_samples_phase1"]),
        n_samples_final=int(CONFIG["n_samples_final"]),
        random_seed=int(CONFIG["random_seed"]),
    )
    seed_instances_df = pd.read_json(CONFIG["seed_instances_input_path"])
    input_messages = (
        sampled_reddit_df["input"].tolist() + seed_instances_df["input"].tolist()
    )
