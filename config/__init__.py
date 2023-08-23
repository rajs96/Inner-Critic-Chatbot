"""Define configuration object"""
from os.path import abspath
import configparser

CONFIG = configparser.ConfigParser()
CONFIG.read(abspath("config/config.ini"))

SCENARIO_GENERATION_CONFIG = CONFIG["scenario_generation"]
SEED_INSTANCE_GENERATION_CONFIG = CONFIG["seed_instance_generation"]
