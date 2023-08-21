"""Define Chat Models used for generation"""
from os import environ
from langchain.chat_models import ChatOpenAI
from config import GPT4_CONFIG


OPENAI_CHAT_MODEL_GPT4 = ChatOpenAI(
    model_name="gpt-4",
    openai_api_key=environ["OPENAI_API_KEY"],
    temperature=float(GPT4_CONFIG["temperature"]),
    top_p=float(GPT4_CONFIG["top_p"]),
    max_tokens=int(GPT4_CONFIG["max_tokens"]),
)
