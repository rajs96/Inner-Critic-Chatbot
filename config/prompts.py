"""Define langchain prompt template objects"""
from config import (
    SCENARIO_GENERATION_CONFIG,
    SEED_INSTANCE_GENERATION_CONFIG,
    INSTRUCTION_FINETUNING_GENERATION_CONFIG,
)
from utils.prompts import create_chat_template
from utils.misc import read_text_file

SCENARIO_GENERATION_PROMPT = create_chat_template(
    system_message_string=read_text_file(
        SCENARIO_GENERATION_CONFIG["system_message_path"]
    ),
    system_message_input_variables=["n_scenarios"],
    input_message_string=read_text_file(
        SCENARIO_GENERATION_CONFIG["input_message_path"]
    ),
    input_message_input_variables=["topic"],
)

SEED_INSTANCE_GENERATION_PROMPT = create_chat_template(
    system_message_string=read_text_file(
        SEED_INSTANCE_GENERATION_CONFIG["system_message_path"]
    ),
    system_message_input_variables=[],
    input_message_string=read_text_file(
        SEED_INSTANCE_GENERATION_CONFIG["input_message_path"]
    ),
    input_message_input_variables=[
        "scenario",
        "theme",
        "emotion",
        "num_sentences",
        "sentence_diversity_level",
    ],
)

INSTRUCTION_FINETUNING_GENERATION_PROMPT = create_chat_template(
    system_message_string=read_text_file(
        INSTRUCTION_FINETUNING_GENERATION_CONFIG["system_message_path"]
    ),
    system_message_input_variables=[],
    input_message_string=read_text_file(
        INSTRUCTION_FINETUNING_GENERATION_CONFIG["input_message_path"]
    ),
    input_message_input_variables=["input_message"],
)
