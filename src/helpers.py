from itertools import zip_longest, islice
import json
import os
import shutil

import yaml


def load_yaml(filename: str):
    with open(filename, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)


def makedirs(path: str, reset: bool = False):
    if os.path.isdir(path):
        if reset:
            shutil.rmtree(path)
            os.makedirs(path)
    else:
        os.makedirs(path)


def save_json_file(filepath, dictionary):
    with open(filepath, "w") as f:
        json.dump(dictionary, f, indent=4, sort_keys=True)
