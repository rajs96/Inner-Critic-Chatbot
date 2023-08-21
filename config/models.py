"""Define Chat Models used for generation"""
from os import environ
from langchain.chat_models import ChatOpenAI
from config import CONFIG

OPENAI_CHAT_MODEL_GPT4 = ChatOpenAI(
    model_name="gpt-4",
    openai_api_key=environ["OPENAI_API_KEY"],
    temperature=CONFIG["openai_gpt4_kwargs"]["temperature"],
    top_p=CONFIG["openai_gpt4_kwargs"]["top_p"],
    max_tokens=CONFIG["openai_gpt4_kwargs"]["max_tokens"],
)
