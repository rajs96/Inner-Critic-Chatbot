"""Supervised finetuning for """
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
import argparse

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--train_data_s3_path", type=str, default=None, help="S3 path to training data"
    # )
    # parser.add_argument(
    #     "--train_data_local_path", type=str, default=None, help="Local path to training data"
    # )
    # parser.add_argument(
    #     "--model_name", type=str, default=None, help="Name of the model to finetune"
    # )
    pass
