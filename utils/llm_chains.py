from langchain import LLMChain
from langchain.llms import BaseLLM
from langchain.prompts import BasePromptTemplate

def create_basic_llm_chain(llm: BaseLLM, prompt: BasePromptTemplate) -> LLMChain:
    return LLMChain(llm=llm, prompt=prompt)