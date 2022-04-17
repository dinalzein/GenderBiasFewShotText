import collections

import numpy as np
import random
from typing import List
from utils.data import get_tsv_data
import torch


def random_sample_cls(sentences: List[str], labels: List[str], n_support: int, n_query: int, label: str):
    """
    Randomly samples Ns examples as support set and Nq as Query set
    """
    data = [sentences[i] for i, lab in enumerate(labels) if lab == label]
    perm = torch.randperm(len(data))
    idx = perm[:n_support]
    support = [data[i] for i in idx]
    idx = perm[n_support: n_support + n_query]
    query = [data[i] for i in idx]

    return support, query


def create_episode(data_dict, n_support, n_classes, n_query, n_unlabeled=0, n_augment=0):
    n_classes = min(n_classes, len(data_dict.keys()))
    rand_keys = np.random.choice(list(data_dict.keys()), n_classes, replace=False)

    assert min([len(val) for val in data_dict.values()]) >= n_support + n_query + n_unlabeled

    for key, val in data_dict.items():
        random.shuffle(val)

    episode = {
        "xs": [
            [data_dict[k][i] for i in range(n_support)] for k in rand_keys
        ],
        "xq": [
            [data_dict[k][n_support + i] for i in range(n_query)] for k in rand_keys
        ]
    }

    if n_unlabeled:
        episode['xu'] = [
            item for k in rand_keys for item in data_dict[k][n_support + n_query:n_support + n_query + 10]
        ]

    if n_augment:
        augmentations = list()
        already_done = list()
        for i in range(n_augment):
            # Draw a random label
            key = random.choice(list(data_dict.keys()))
            # Draw a random data index
            ix = random.choice(range(len(data_dict[key])))
            # If already used, re-sample
            while (key, ix) in already_done:
                key = random.choice(list(data_dict.keys()))
                ix = random.choice(range(len(data_dict[key])))
            already_done.append((key, ix))
            if "augmentations" not in data_dict[key][ix]:
                raise KeyError(f"Input data {data_dict[key][ix]} does not contain any augmentations / is not properly formatted.")
            augmentations.append((
                data_dict[key][ix]["sentence"],
                [item["text"] for item in data_dict[key][ix]["augmentations"]]
            ))
        episode["x_augment"] = augmentations

    return episode

def create_gender_balance_episode(data_dict, n_support, n_classes, n_query, n_unlabeled=0, n_augment=0):
    gender_keys=['F', 'M']
    n_classes = min(n_classes, len(data_dict.keys()))
    rand_keys = np.random.choice(list(data_dict.keys()), n_classes, replace=False)
    #assert min([len(val) for val in data_dict[key].values() for key in data_dict.keys()]) >= (n_support + n_query + n_unlabeled)/2
    assert min([ len(val) for key in data_dict.keys() for val in data_dict[key].values()]) >= (n_support + n_query + n_unlabeled)/2

    assert n_support %2==0 and n_query%2==0
    for key, val in data_dict.items():
        for key2, val2 in data_dict[key].items():
            random.shuffle(val2)
    episode = {
        "xs": [
            [data_dict[k][j][i] for i in range(int(n_support/2)) for j in gender_keys] for k in rand_keys
        ],
        "xq": [
            [data_dict[k][j][int(n_support/2) + i] for i in range(int(n_query/2)) for j in gender_keys] for k in rand_keys
        ]
    }
    if n_unlabeled:
        episode['xu'] = [
            item for k in rand_keys for j in gender_keys for item in data_dict[k][j][n_support + n_query:n_support + n_query + 1]
        ]
    return episode


def create_ARSC_train_episode(prefix: str = "data/ARSC-Yu/raw", n_support: int = 5, n_query: int = 5, n_unlabeled=0):
    labels = sorted(
        set([line.strip() for line in open(f"{prefix}/workspace.filtered.list", "r").readlines()])
        - set([line.strip() for line in open(f"{prefix}/workspace.target.list", "r").readlines()]))

    # Pick a random label
    label = random.choice(labels)

    # Pick a random binary task (2, 4, 5)
    binary_task = random.choice([2, 4, 5])

    # Fix: this label/binary task sucks
    while label == "office_products" and binary_task == 2:
        # Pick a random label
        label = random.choice(labels)

        # Pick a random binary task (2, 4, 5)
        binary_task = random.choice([2, 4, 5])

    data = (
            get_tsv_data(f"{prefix}/{label}.t{binary_task}.train", label=label) +
            get_tsv_data(f"{prefix}/{label}.t{binary_task}.dev", label=label) +
            get_tsv_data(f"{prefix}/{label}.t{binary_task}.test", label=label)
    )

    random.shuffle(data)
    task = collections.defaultdict(list)
    for d in data:
        task[d['label']].append(d['sentence'])
    task = dict(task)

    assert min([len(val) for val in task.values()]) >= n_support + n_query + n_unlabeled, \
        f"Label {label}_{binary_task}: min samples is {min([len(val) for val in task.values()])} while K+Q+U={n_support + n_query + n_unlabeled}"

    for key, val in task.items():
        random.shuffle(val)

    episode = {
        "xs": [
            [task[k][i] for i in range(n_support)] for k in task.keys()
        ],
        "xq": [
            [task[k][n_support + i] for i in range(n_query)] for k in task.keys()
        ]
    }

    if n_unlabeled:
        episode['xu'] = [
            item for k in task.keys() for item in task[k][n_support + n_query:n_support + n_query + n_unlabeled]
        ]
    return episode


def create_ARSC_test_episode(prefix: str = "data/ARSC-Yu/raw", n_query: int = 5, n_unlabeled=0, set_type: str = "test"):
    assert set_type in ("test", "dev")
    labels = [line.strip() for line in open(f"{prefix}/workspace.target.list", "r").readlines()]

    # Pick a random label
    label = random.choice(labels)

    # Pick a random binary task (2, 4, 5)
    binary_task = random.choice([2, 4, 5])

    support_data = get_tsv_data(f"{prefix}/{label}.t{binary_task}.train", label=label)
    assert len(support_data) == 10  # 2 * 5 shots
    support_dict = collections.defaultdict(list)
    for d in support_data:
        support_dict[d['label']].append(d['sentence'])

    query_data = get_tsv_data(f"data/ARSC-Yu/raw/{label}.t{binary_task}.{set_type}", label=label)
    query_dict = collections.defaultdict(list)
    for d in query_data:
        query_dict[d['label']].append(d['sentence'])

    assert min([len(val) for val in query_dict.values()]) >= n_query + n_unlabeled

    for key, val in query_dict.items():
        random.shuffle(val)

    episode = {
        "xs": [
            [sentence for sentence in support_dict[k]] for k in sorted(query_dict.keys())
        ],
        "xq": [
            [query_dict[k][i] for i in range(n_query)] for k in sorted(query_dict.keys())
        ]
    }

    if n_unlabeled:
        episode['xu'] = [
            item for k in sorted(query_dict.keys()) for item in query_dict[k][n_query:n_query + n_unlabeled]
        ]
    return episode
