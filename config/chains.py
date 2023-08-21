"""Define LLM chains"""
from langchain import LLMChain

from config.models import OPENAI_CHAT_MODEL_GPT4
from config.prompts import SCENARIO_GENERATION_PROMPT

SCENARIO_GENERATION_CHAIN = LLMChain(
    llm=OPENAI_CHAT_MODEL_GPT4, prompt=SCENARIO_GENERATION_PROMPT
)
