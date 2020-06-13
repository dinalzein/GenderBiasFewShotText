import json
import argparse

import tqdm

from models.encoders.bert_encoder import BERTEncoder
from utils.data import get_jsonl_data
from utils.python import now, set_seeds
import random
import collections
import os
from typing import List, Dict
from tensorboardX import SummaryWriter
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as torch_functional
from torch.autograd import Variable
import warnings
import logging
from utils.few_shot import create_episode
from utils.math import euclidean_dist, cosine_similarity

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

warnings.simplefilter('ignore')

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class BaselineNet(nn.Module):
    def __init__(
            self,
            encoder,
            is_pp: bool = False,
            hidden_dim: int = 768
    ):
        super(BaselineNet, self).__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(p=0.25).to(device)
        self.is_pp = is_pp
        self.hidden_dim = hidden_dim

    def train_model(
            self,
            data_dict: Dict[str, List[str]],
            summary_writer: SummaryWriter = None,
            n_epoch: int = 400,
            batch_size: int = 16,
            log_every: int = 10):
        self.train()

        training_classes = sorted(set(data_dict.keys()))
        n_training_classes = len(training_classes)
        class_to_ix = {c: ix for ix, c in enumerate(training_classes)}
        training_data_list = [{"sentence": sentence, "label": label} for label, sentences in data_dict.items() for sentence in sentences]

        training_matrix = None
        training_classifier = None

        if self.is_pp:
            training_matrix = torch.nn.Parameter(torch.randn(n_training_classes, self.hidden_dim))
            optimizer = torch.optim.Adam(list(self.parameters()) + [training_matrix], lr=2e-5)
        else:
            training_classifier = nn.Linear(in_features=self.hidden_dim, out_features=n_training_classes).to(device)
            optimizer = torch.optim.Adam(list(self.parameters()) + list(training_classifier.parameters()), lr=2e-5)

        n_samples = len(training_data_list)
        loss_fn = nn.CrossEntropyLoss()
        global_step = 0

        # Metrics
        training_losses = list()
        training_accuracies = list()

        for _ in tqdm.tqdm(range(n_epoch)):
            random.shuffle(training_data_list)
            for ix in tqdm.tqdm(range(0, n_samples, batch_size)):
                optimizer.zero_grad()
                torch.cuda.empty_cache()

                batch_items = training_data_list[ix:ix + batch_size]
                batch_sentences = [d['sentence'] for d in batch_items]
                batch_labels = torch.Tensor([class_to_ix[d['label']] for d in batch_items]).long().to(device)
                z = self.encoder(batch_sentences)
                if self.is_pp:
                    z = cosine_similarity(z, training_matrix) * 5
                else:
                    z = self.dropout(z)
                    z = training_classifier(z)
                loss = loss_fn(input=z, target=batch_labels)
                acc = (z.argmax(1) == batch_labels).float().mean()
                loss.backward()
                optimizer.step()

                global_step += 1
                training_losses.append(loss.item())
                training_accuracies.append(acc.item())
                if (global_step % log_every) == 0:
                    if summary_writer:
                        summary_writer.add_scalar(tag="loss", global_step=global_step, scalar_value=np.mean(training_losses))
                        summary_writer.add_scalar(tag="acc", global_step=global_step, scalar_value=np.mean(training_accuracies))
                    # Empty metrics
                    training_losses = list()
                    training_accuracies = list()

    def test_one_episode(
            self,
            support_data_dict: Dict[str, List[str]],
            query_data_dict: Dict[str, List[str]],
            batch_size: int = 4,
            n_iter: int = 100,
            summary_writer: SummaryWriter = None,
            summary_tag_prefix: str = None):

        # Check data integrity
        assert set(support_data_dict.keys()) == set(query_data_dict.keys())

        # Freeze encoder
        self.encoder.eval()

        episode_classes = sorted(set(support_data_dict.keys()))
        n_episode_classes = len(episode_classes)
        class_to_ix = {c: ix for ix, c in enumerate(episode_classes)}
        support_data_list = [{"sentence": sentence, "label": label} for label, sentences in support_data_dict.items() for sentence in sentences]
        support_data_list = (support_data_list * 400)[:400]

        loss_fn = nn.CrossEntropyLoss()
        episode_matrix = None
        episode_classifier = None
        if self.is_pp:
            episode_matrix = torch.nn.Parameter(torch.randn(n_episode_classes, self.hidden_dim))
            optimizer = torch.optim.Adam(list(self.parameters()) + [episode_matrix], lr=1e-3)
        else:
            episode_classifier = nn.Linear(in_features=self.hidden_dim, out_features=n_episode_classes).to(device)
            optimizer = torch.optim.Adam(list(self.parameters()) + list(episode_classifier.parameters()), lr=1e-3)

            # Train on support
        for iteration in range(n_iter):
            optimizer.zero_grad()
            torch.cuda.empty_cache()

            batch = support_data_list[iteration * batch_size: iteration * batch_size + batch_size]
            batch_sentences = [d['sentence'] for d in batch]
            batch_labels = torch.Tensor([class_to_ix[d['label']] for d in batch]).long().to(device)
            z = self.encoder(batch_sentences)

            if self.is_pp:
                z = cosine_similarity(z, episode_matrix) * 5
            else:
                z = self.dropout(z)
                z = episode_classifier(z)

            loss = loss_fn(input=z, target=batch_labels)
            acc = (z.argmax(1) == batch_labels).float().mean()
            loss.backward()
            optimizer.step()

            if summary_writer:
                summary_writer.add_scalar(tag=f'{summary_tag_prefix}_loss', global_step=iteration, scalar_value=loss.item())
                summary_writer.add_scalar(tag=f'{summary_tag_prefix}_acc', global_step=iteration, scalar_value=acc.item())

        # Predict on query
        self.eval()
        episode_classifier.eval()
        query_data_list = [{"sentence": sentence, "label": label} for label, sentences in query_data_dict.items() for sentence in sentences]
        query_labels = torch.Tensor([class_to_ix[d['label']] for d in query_data_list]).long().to(device)
        logits = list()
        with torch.no_grad():
            for ix in range(0, len(query_data_list), 16):
                batch = query_data_list[ix:ix + 16]
                batch_sentences = [d['sentence'] for d in batch]
                z = self.encoder(batch_sentences)

                if self.is_pp:
                    z = cosine_similarity(z, episode_matrix) * 5
                else:
                    z = episode_classifier(z)

                logits.append(z)
        logits = torch.cat(logits, dim=0)
        y_hat = logits.argmax(1)

        loss = loss_fn(input=logits, target=query_labels)
        acc = (y_hat == query_labels).float().mean()

        return {
            "loss": loss.item(),
            "acc": acc.item()
        }

    def test_model(
            self,
            data_dict: Dict[str, List[str]],
            n_support: int,
            n_classes: int,
            n_episodes=600,
            summary_writer: SummaryWriter = None
    ):
        test_metrics = list()

        for episode in tqdm.tqdm(range(n_episodes)):
            episode_classes = np.random.choice(list(data_dict.keys()), size=n_classes, replace=False)
            episode_query_data_dict = dict()
            episode_support_data_dict = dict()

            for episode_class in episode_classes:
                random.shuffle(data_dict[episode_class])
                episode_support_data_dict[episode_class] = data_dict[episode_class][:n_support]
                episode_query_data_dict[episode_class] = data_dict[episode_class][n_support:]

            episode_metrics = self.test_one_episode(
                support_data_dict=episode_support_data_dict,
                query_data_dict=episode_query_data_dict,

            )
            test_metrics.append(episode_metrics)
            for metric_name, metric_value in episode_metrics.items():
                summary_writer.add_scalar(tag=metric_name, global_step=episode, scalar_value=metric_value)

        return test_metrics


def run_baseline(
        train_path: str,
        model_name_or_path: str,
        n_support: int,
        n_classes: int,
        valid_path: str = None,
        test_path: str = None,
        output_path: str = f'runs/{now()}',
        n_test_episodes: int = 600,
        log_every: int = 10,
        n_train_epoch: int = 400,
        train_batch_size: int = 16
):
    if output_path:
        if os.path.exists(output_path):
            raise FileExistsError(f"Output path {output_path} already exists. Exiting.")

    # --------------------
    # Creating Log Writers
    # --------------------
    os.makedirs(output_path)
    os.makedirs(os.path.join(output_path, "logs/train"))
    train_writer: SummaryWriter = SummaryWriter(logdir=os.path.join(output_path, "logs/train"), flush_secs=1, max_queue=1)
    valid_writer: SummaryWriter = None
    test_writer: SummaryWriter = None
    log_dict = dict(train=list())

    if valid_path:
        os.makedirs(os.path.join(output_path, "logs/valid"))
        valid_writer = SummaryWriter(logdir=os.path.join(output_path, "logs/valid"), flush_secs=1, max_queue=1)
        log_dict["valid"] = list()
    if test_path:
        os.makedirs(os.path.join(output_path, "logs/test"))
        test_writer = SummaryWriter(logdir=os.path.join(output_path, "logs/test"), flush_secs=1, max_queue=1)
        log_dict["test"] = list()

    def raw_data_to_labels_dict(data, shuffle=True):
        labels_dict = collections.defaultdict(list)
        for item in data:
            labels_dict[item['label']].append(item["sentence"])
        labels_dict = dict(labels_dict)
        if shuffle:
            for key, val in labels_dict.items():
                random.shuffle(val)
        return labels_dict

    # Load model
    bert = BERTEncoder(model_name_or_path).to(device)
    baseline_net = BaselineNet(encoder=bert).to(device)

    # Load data
    train_data = get_jsonl_data(train_path)
    train_data_dict = raw_data_to_labels_dict(train_data, shuffle=True)
    logger.info(f"train labels: {train_data_dict.keys()}")

    if valid_path:
        valid_data = get_jsonl_data(valid_path)
        valid_data_dict = raw_data_to_labels_dict(valid_data, shuffle=True)
        logger.info(f"valid labels: {valid_data_dict.keys()}")
    else:
        valid_data_dict = None

    if test_path:
        test_data = get_jsonl_data(test_path)
        test_data_dict = raw_data_to_labels_dict(test_data, shuffle=True)
        logger.info(f"test labels: {test_data_dict.keys()}")
    else:
        test_data_dict = None

    baseline_net.train_model(
        data_dict=train_data_dict,
        summary_writer=train_writer,
        n_epoch=n_train_epoch,
        batch_size=train_batch_size,
        log_every=log_every
    )

    # Validation
    validation_metrics = baseline_net.test_model(
        data_dict=valid_data_dict,
        n_support=n_support,
        n_classes=n_classes,
        n_episodes=n_test_episodes,
        summary_writer=valid_writer
    )

    # Test
    test_metrics = baseline_net.test_model(
        data_dict=test_data_dict,
        n_support=n_support,
        n_classes=n_classes,
        n_episodes=n_test_episodes,
        summary_writer=test_writer
    )

    with open(os.path.join(output_path, 'validation_metrics.json'), "w") as file:
        json.dump(validation_metrics, file, ensure_ascii=False)
    with open(os.path.join(output_path, 'test_metrics.json'), "w") as file:
        json.dump(test_metrics, file, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", type=str, required=True, help="Path to training data")
    parser.add_argument("--valid-path", type=str, default=None, help="Path to validation data")
    parser.add_argument("--test-path", type=str, default=None, help="Path to testing data")

    parser.add_argument("--output-path", type=str, default=f'runs/{now()}')
    parser.add_argument("--model-name-or-path", type=str, required=True, help="Transformer model to use")
    parser.add_argument("--log-every", type=int, default=10, help="Number of training episodes between each logging")
    parser.add_argument("--seed", type=int, default=42, help="Random seed to set")

    # Few-Shot related stuff
    parser.add_argument("--n-support", type=int, help="Number of support points for each class", required=True)
    parser.add_argument("--n-classes", type=int, help="Number of classes per episode", required=True)
    parser.add_argument("--n-test-episodes", type=int, default=600, help="Number of episodes during evaluation (valid, test)")
    parser.add_argument("--n-train-epoch", type=int, default=400, help="Number of epoch during training")
    parser.add_argument("--train-batch-size", type=int, default=16, help="Batch size used during training")

    args = parser.parse_args()

    # Set random seed
    set_seeds(args.seed)

    # Check if data path(s) exist
    for arg in [args.train_path, args.valid_path, args.test_path]:
        if arg and not os.path.exists(arg):
            raise FileNotFoundError(f"Data @ {arg} not found.")

    # Run
    run_baseline(
        train_path=args.train_path,
        valid_path=args.valid_path,
        test_path=args.test_path,
        output_path=args.output_path,

        model_name_or_path=args.model_name_or_path,

        n_support=args.n_support,
        n_classes=args.n_classes,
        n_test_episodes=args.n_test_episodes,
        log_every=args.log_every,
        n_train_epoch=args.n_train_epoch,
        train_batch_size=args.train_batch_size
    )

    # Save config
    with open(os.path.join(args.output_path, "config.json"), "w") as file:
        json.dump(vars(args), file, ensure_ascii=False)


if __name__ == '__main__':
    main()
