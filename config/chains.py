"""Define LLM chains"""
from langchain import LLMChain

from config.models import OPENAI_CHAT_MODEL_GPT4
from config.prompts import TOPIC_GENERATION_PROMPT

TOPIC_GENERATION_CHAIN = LLMChain(
    llm=OPENAI_CHAT_MODEL_GPT4, prompt=TOPIC_GENERATION_PROMPT
)
