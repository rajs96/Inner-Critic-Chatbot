"""For synthetic training data, use an LLM to generate scenarios for a given topic"""
import time
from os.path import abspath
from typing import List
from tqdm import tqdm

from config import SCENARIO_GENERATION_CONFIG
from config.chains import SCENARIO_GENERATION_CHAIN
from utils.misc import read_text_file_lines


def extract_scenarios_by_topic(topic: str, n_scenarios: int) -> str:
    """For a particular topic, generate scenarios using prompt and GPT4"""
    return SCENARIO_GENERATION_CHAIN.run(topic=topic, n_scenarios=n_scenarios)


def generate_all_scenarios(
    topics: List[str], n_scenarios_per_topic: int, batch_size: int
) -> List[str]:
    """Generate scenarios for all topics"""
    scenario_strings = []
    for topic in tqdm(topics, desc="Processing each topic"):
        print(f"Processing topic: {topic}")
        for _ in tqdm(
            range(0, n_scenarios_per_topic, batch_size),
            desc="Processing scenario batches",
        ):
            scenario_string = extract_scenarios_by_topic(
                topic=topic, n_scenarios=batch_size
            )
            scenario_strings.append(scenario_string)
            # Give buffer due to GPT4 rate limits
            time.sleep(3)

    # Answer is in the form of bullet points, need to extract
    scenarios = [
        line.strip()[2:]
        for scenarios_string in scenario_strings
        for line in scenarios_string.splitlines()
        if line.strip().startswith("-")
    ]
    return scenarios


if __name__ == "__main__":
    # Get topics, number of scenarios, and batch size from the predefined config
    all_topics = read_text_file_lines(
        abspath(SCENARIO_GENERATION_CONFIG["input_topics_path"])
    )
    n_scenarios_per_topic = int(SCENARIO_GENERATION_CONFIG["n_scenarios_per_topic"])
    batch_size = int(SCENARIO_GENERATION_CONFIG["generation_batch_size"])

    # Generate scenarios and write to text file
    all_scenarios = generate_all_scenarios(
        all_topics, n_scenarios_per_topic, batch_size
    )
    output_path = abspath(SCENARIO_GENERATION_CONFIG["generated_scenarios_path"])
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(all_scenarios))
