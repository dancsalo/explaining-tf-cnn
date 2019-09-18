import click
import logging
import sys
import os

from easydict import EasyDict as edict
import yaml

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.getcwd()))

from src import ROOT_DIR
from src.helpers import load_yaml
from src.model import train_from_yaml


@click.command()
@click.option("--yaml", default="v1", help="Name of yaml file to train")
def train(yaml):
    yaml_path = os.path.join(ROOT_DIR, "scripts", "train", yaml + ".yaml")
    FLAGS = edict(load_yaml(yaml_path))
    train_from_yaml(FLAGS)


if __name__ == "__main__":
    train()
