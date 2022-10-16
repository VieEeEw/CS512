import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from typing import List, Tuple
import os

CAR_CSV = "car_data.csv"
DIR = "HW1_source"
VALIDATION_RATIO = 0.8


def parse_csv() -> Tuple[np.ndarray, np.ndarray, int, int]:
    with open(os.path.join(DIR, CAR_CSV), 'r') as f:
        f.readline()
        age_max = 0
        salary_max = 0
        features: List[List[int]] = []
        labels: List[int] = []
        for line in f:
            if not line:
                break
            _, _, age, salary, purchased = line.strip().split(",")
            age = int(age)
            salary = int(salary)
            purchased = int(purchased)
            features.append([age, salary])
            labels.append(purchased)
            age_max = max(age_max, age)
            salary_max = max(salary_max, salary)
        return np.array(features, dtype=np.int32), np.array(labels, dtype=np.int32), age, salary_max


def normalize(features: np.ndarray, age_max: int, salary_max: int) -> np.ndarray:
    ret = features.astype(np.float64)
    ret[:, 0] /= age_max
    ret[:, 1] /= salary_max
    return ret


def split(features: np.ndarray, labels: np.ndarray) -> Tuple[
    Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    threshold = int(features.shape[0] * VALIDATION_RATIO)
    return (features[: threshold, :], labels[:threshold]), (features[threshold:], labels[threshold:])


def eval_model(model, val_ds, test_ds):
    model.fit(*val_ds)
    test_features, test_labels = test_ds
    print(accuracy_score(test_labels, model.predict(test_features)))


def report_with_norm(val_ds: Tuple[np.ndarray, np.ndarray], test_ds: Tuple[np.ndarray, np.ndarray], norm=True):
    max_iter = -1 if norm else 500
    linear = SVC(kernel="linear", max_iter=max_iter)
    logistic = LogisticRegression(solver="liblinear", max_iter=500)
    kernel = SVC(gamma=1, max_iter=max_iter)
    eval_model(linear, val_ds, test_ds)
    eval_model(logistic, val_ds, test_ds)
    eval_model(kernel, val_ds, test_ds)


if __name__ == "__main__":
    f, label, f1_max, f2_max = parse_csv()
    val, test = split(f, label)
    f_norm = normalize(f, f1_max, f2_max)
    val_norm, test_norm = split(f_norm, label)
    print("Normed")
    report_with_norm(val_norm, test_norm)
    print("Not normed")
    report_with_norm(val, test, norm=False)
