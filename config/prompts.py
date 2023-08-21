"""Define langchain prompt template objects"""
from os.path import abspath
from langchain.prompts import (
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from config import CONFIG
from utils.prompts import create_chat_instruct_template
from utils.misc import read_text_file

SCENARIO_GENERATION_SYSTEM_MESSAGE = SystemMessagePromptTemplate(
    prompt=PromptTemplate(
        template=read_text_file(
            abspath(CONFIG["scenario_generation"]["system_message_path"])
        ),
        input_variables=["n_scenarios"],
    )
)
SCENARIO_GENERATION_INPUT_MESSAGE = HumanMessagePromptTemplate(
    prompt=PromptTemplate(
        template=read_text_file(
            abspath(CONFIG["scenario_generation"]["input_message_path"])
        ),
        input_variables=["topic"],
    )
)
SCENARIO_GENERATION_PROMPT = create_chat_instruct_template(
    SCENARIO_GENERATION_SYSTEM_MESSAGE, SCENARIO_GENERATION_INPUT_MESSAGE
)
