"""Define Chat Models used for generation"""
from config import SCENARIO_GENERATION_CONFIG, SEED_INSTANCE_GENERATION_CONFIG
from utils.models import create_chat_model

SCENARIO_GENERATION_MODEL = create_chat_model(SCENARIO_GENERATION_CONFIG)
SEED_INSTANCE_GENERATION_MODEL = create_chat_model(SEED_INSTANCE_GENERATION_CONFIG)
