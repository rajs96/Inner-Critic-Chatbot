from langchain.schema import SystemMessage
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)

def create_chat_instruct_template(
    system_prompt: SystemMessage, user_prompt: HumanMessagePromptTemplate
) -> ChatPromptTemplate:
    """Chat template used primarily for instructing an LLM to generate data"""
    return ChatPromptTemplate(messages=[system_prompt, user_prompt])
