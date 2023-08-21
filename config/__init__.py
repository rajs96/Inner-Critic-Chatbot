"""Define configuration object"""
from os.path import abspath
import configparser

CONFIG = configparser.ConfigParser()
CONFIG.read(abspath("config/config.ini"))

GPT4_CONFIG = CONFIG["openai_gpt4_kwargs"]
SCENARIO_GENERATION_CONFIG = CONFIG["scenario_generation"]
