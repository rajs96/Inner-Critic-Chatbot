"""Define configuration object"""
from os.path import abspath
import configparser

CONFIG = configparser.ConfigParser()
CONFIG.read(abspath("config/config.ini"))
