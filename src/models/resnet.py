import argparse
from model_exec_helper import BaseArgumentParser
from models import ResnetModelCreator
from models import ResnetModel
import os
import pathlib
import sys

import torch

def get_best_resnet_available_and_transforms(device:str = "cuda"):
    model_creator = ResnetModelCreator()

    resnet_model = model_creator.factory_method(device)

    return resnet_model, resnet_model.preprocess

def get_arg_parser(parent_parser: argparse.ArgumentParser = None) -> BaseArgumentParser:
    parser = BaseArgumentParser(parent_parser)
    return parser

if __name__ == "__main__":

    parent_dir = pathlib.Path(__file__).parent
    SCRIPT_DIR = os.path.dirname(os.path.abspath(parent_dir))
    sys.path.append(os.path.dirname(SCRIPT_DIR))
    from src import transforms

    print(f"PyTorch version: {torch.__version__}")

    model, preprocess = get_best_resnet_available_and_transforms()

    print("deu certo")
    parser = get_arg_parser()

    user_args = parser.parse_args()