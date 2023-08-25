"""Define configuration object"""
from os.path import dirname, abspath, join
import configparser

CONFIG = configparser.ConfigParser()
CUR_DIR = dirname(abspath(__file__))  # gives abs path of the directory this file is in
BASE_DIR = dirname(CUR_DIR)  # this always gives the correct working directory
CONFIG.read(join(BASE_DIR, "config/config.ini"))

# Config defines relative paths, add absolute paths to the config
for section in CONFIG.sections():
    for key, value in CONFIG[section].items():
        if "path" in key:
            absolute_path = join(BASE_DIR, value)
            CONFIG.set(section, key, absolute_path)

SCENARIO_GENERATION_CONFIG = CONFIG["scenario_generation"]
SEED_INSTANCE_GENERATION_CONFIG = CONFIG["seed_instance_generation"]
