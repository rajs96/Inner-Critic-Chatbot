from typing import Dict, Union
from os.path import join
import pandas as pd
import random
import ast
from config import BASE_DIR
from utils.misc import read_text_file, write_text_file


def extract_example(row: pd.Series, theme: str) -> str:
    def check_dict(s: str) -> Union[Dict[str, str], str]:
        try:
            dict_obj = ast.literal_eval(s)
            return ", ".join(dict_obj["text"])
        except (SyntaxError, ValueError):
            return s

    def extract_section_from_theme(theme: str) -> str:
        return f"""
        {theme} Identification: {check_dict(row[f"{theme}Id"])}
        Instruction: {check_dict(row[f"{theme}Instruction"])}
        Instruction Goal: {check_dict(row[f"{theme}InstructionGoal"])}
        """

    res = ""
    res += f"[Input]: {row.input} \n"
    res += "[Response]: \n"
    res += f"Theme: {theme} \n"
    res += extract_section_from_theme(theme) + "\n"

    return res


if __name__ == "__main__":
    FEW_SHOT_EXAMPLES_PATH = join(
        BASE_DIR, "data/training_data/assets/few_shot_examples.csv"
    )
    RANDOM_SEED = 42
    random.seed(RANDOM_SEED)
    # take 10 in-context examples
    FEW_SHOT_EXAMPLES_DF = pd.read_csv(FEW_SHOT_EXAMPLES_PATH, index_col=0)
    few_shot_examples = []
    for i, row in FEW_SHOT_EXAMPLES_DF.iterrows():
        for theme in ["desire", "fear", "emotion", "selfBelief"]:
            example = extract_example(row, theme)
            few_shot_examples.append(example)

    # take random 10 examples
    random.shuffle(few_shot_examples)
    FEW_SHOT_EXAMPLES_STR = " ".join(few_shot_examples[:10])
    SYSTEM_MESSAGE_BASE = read_text_file(
        join(
            BASE_DIR,
            "prompts/system_messages/instruction_finetuning_system_message_base.txt",
        )
    )
    SYSTEM_MESSAGE = SYSTEM_MESSAGE_BASE + "\n" + FEW_SHOT_EXAMPLES_STR
    write_text_file(
        join(
            BASE_DIR,
            "prompts/system_messages/instruction_finetuning_system_message.txt",
        ),
        SYSTEM_MESSAGE,
    )
