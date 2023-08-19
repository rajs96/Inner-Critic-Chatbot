"""Define langchain prompt template objects"""
import os
from langchain.schema import SystemMessage
from langchain.prompts import HumanMessagePromptTemplate

from config import CONFIG
from utils.prompts import create_chat_instruct_template
from utils.misc import read_text_file

TOPIC_GENERATION_SYSTEM_MESSAGE = SystemMessage(
    content=read_text_file(
        os.path.abspath(CONFIG["local_paths"]["topic_generation_input_message"])
    )
)
TOPIC_GENERATION_INPUT_MESSAGE = HumanMessagePromptTemplate(
    content=read_text_file(
        os.path.abspath(CONFIG["local_paths"]["topic_generation_system_message"])
    ),
    input_variables=["topic"],
)
TOPIC_GENERATION_PROMPT = create_chat_instruct_template(
    TOPIC_GENERATION_SYSTEM_MESSAGE, TOPIC_GENERATION_INPUT_MESSAGE
)
