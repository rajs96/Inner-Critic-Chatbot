"""Define langchain prompt template objects"""
from os.path import abspath
from langchain.schema import SystemMessage
from langchain.prompts import PromptTemplate, HumanMessagePromptTemplate

from config import CONFIG
from utils.prompts import create_chat_instruct_template
from utils.misc import read_text_file

TOPIC_GENERATION_SYSTEM_MESSAGE = SystemMessage(
    content=read_text_file(
        abspath(CONFIG["local_paths"]["topic_generation_system_message"])
    )
)
TOPIC_GENERATION_INPUT_MESSAGE = HumanMessagePromptTemplate(
    prompt=PromptTemplate(
        template=read_text_file(
            abspath(CONFIG["local_paths"]["topic_generation_input_message"])
        ),
        input_variables=["topic"],
    )
)
TOPIC_GENERATION_PROMPT = create_chat_instruct_template(
    TOPIC_GENERATION_SYSTEM_MESSAGE, TOPIC_GENERATION_INPUT_MESSAGE
)
