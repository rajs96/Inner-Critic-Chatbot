"""Generate seed instances from scenarios"""
import random
import asyncio
from tqdm.asyncio import tqdm_asyncio
from os.path import abspath
from typing import List, Dict, Tuple
from config import SEED_INSTANCE_GENERATION_CONFIG as CONFIG
from config.chains import SEED_INSTANCE_GENERATION_CHAIN
from utils.misc import read_text_file_lines, write_json


async def generate_input_instance(
    scenario: str,
    theme: str,
    emotion: str,
    num_sentences: int,
    sentence_diversity_level: str,
) -> str:
    """Generate chatbot input based on relevant variables"""
    return await SEED_INSTANCE_GENERATION_CHAIN.arun(
        scenario=scenario,
        theme=theme,
        emotion=emotion,
        num_sentences=num_sentences,
        sentence_diversity_level=sentence_diversity_level,
    )


async def generate_all_input_instances(
    scenarios: List[str],
    themes: List[str],
    emotions: List[str],
    num_sentences_range: Tuple[int, int],
    sentence_diversity_levels: List[str],
    num_instances_per_scenario: int,
) -> List[Dict[str, str]]:
    records = []
    input_instance_attributes = [
        {
            "scenario": scenario,
            "theme": random.choice(themes),
            "emotion": random.choice(emotions),
            "num_sentences": random.randint(
                num_sentences_range[0], num_sentences_range[1]
            ),
            "sentence_diversity_level": random.choice(sentence_diversity_levels),
        }
        for scenario in scenarios
        for _ in range(num_instances_per_scenario)
    ]
    num_instances = len(input_instance_attributes)
    request_batch_size = 60
    for i in tqdm_asyncio(
        range(0, num_instances, request_batch_size),
        desc="Making OpenAI requests in batches...",
    ):
        tasks = []
        print(f"Started generating instances {i} to {i+request_batch_size}")
        for j in tqdm_asyncio(
            range(i, min(i + request_batch_size, num_instances)),
            desc="Generating instances for batch",
        ):
            tasks.append(generate_input_instance(**input_instance_attributes[j]))
            await asyncio.sleep(0.5)

        batch_records = await asyncio.gather(*tasks)
        batch_records = [
            {"input": input, **input_instance_attributes[j]}
            for j, input in zip(range(i, i + request_batch_size), batch_records)
        ]
        print(f"Completed generating instances {i} to {i+request_batch_size}")
        records.extend(batch_records)
        await asyncio.sleep(5)

    return records


if __name__ == "__main__":
    scenarios = read_text_file_lines(CONFIG["input_scenarios_path"])
    themes = read_text_file_lines(CONFIG["input_themes_path"])
    emotions = read_text_file_lines(CONFIG["input_emotions_path"])
    num_sentences_range = [
        int(CONFIG["num_sentences_min"]),
        int(CONFIG["num_sentences_max"]),
    ]
    sentence_diversity_levels = read_text_file_lines(
        CONFIG["input_diversity_levels_path"]
    )
    num_instances_per_scenario = int(CONFIG["num_instances_per_scenario"])

    instances = asyncio.run(
        generate_all_input_instances(
            scenarios=scenarios,
            themes=themes,
            emotions=emotions,
            num_sentences_range=num_sentences_range,
            sentence_diversity_levels=sentence_diversity_levels,
            num_instances_per_scenario=num_instances_per_scenario,
        )
    )
    write_json(
        instances,
        abspath(CONFIG["seed_instances_output_path"]),
    )
