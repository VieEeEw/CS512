import numpy as np
from typing import Dict, Any, Optional
from pyod.models.lof import LOF
from pyod.models.auto_encoder_torch import AutoEncoder
from sklearn.metrics import f1_score
from glob import glob
import os

DATA_NAME = "ALOI.npz"


def train_report(output_file, feature, label, clf, model_args: Optional[Dict[str, Any]] = None):
    if model_args is None:
        model_args = {}
    model = clf(**model_args)
    model.fit(feature)
    output_file.write(f"Model: {clf.__name__}, F-Score: {f1_score(label, model.predict(feature), average='micro')}\n")


def solve(output_file):
    data_file = glob(DATA_NAME) + glob(os.path.join("*", DATA_NAME))
    if not data_file:
        output_file.write(f"Data file {DATA_NAME} is not found anywhere, please check.\n")
        exit(0)
    data = np.load(data_file[0])
    x = data["X"]
    y = data["y"]
    output_file.write(f"Number of samples: {x.shape[0]}\nNumber of features: {x.shape[1]}\n")
    train_report(output_file, x, y, LOF)
    train_report(output_file, x, y, AutoEncoder, {"epochs": 15, "hidden_neurons": [27, 12, 12, 27]})


if __name__ == "__main__":
    output = "hw2-6-results.txt"
    print(f"Results will be saved in {output}")
    with open(output, "w") as f:
        solve(f)
