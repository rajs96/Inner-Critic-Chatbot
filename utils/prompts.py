"""Prompt utilities"""
from typing import List
from langchain.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)


def create_chat_template(
    system_message_string: str,
    system_message_input_variables: List[str],
    input_message_string: str,
    input_message_input_variables: List[str],
) -> ChatPromptTemplate:
    """Utility function to create chat template"""
    system_message_template = SystemMessagePromptTemplate(
        prompt=PromptTemplate(
            template=system_message_string,
            input_variables=system_message_input_variables,
        ),
    )
    input_message_template = HumanMessagePromptTemplate(
        prompt=PromptTemplate(
            template=input_message_string, input_variables=input_message_input_variables
        ),
    )
    return ChatPromptTemplate(
        messages=[system_message_template, input_message_template]
    )
