from dataset import LifeWatchDataset
import yaml
import pathlib
import numpy as np
import os
import pandas as pd

config_path="config.yaml"
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)


L=LifeWatchDataset(config)
L.train_clap()