from typing import List

import torch.nn as nn
import logging
import warnings
import torch
from transformers import AutoModel, AutoTokenizer

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

warnings.simplefilter('ignore')

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class BERTEncoder(nn.Module):
    def __init__(self, config_name_or_path):
        super(BERTEncoder, self).__init__()
        logger.info(f"Loading Encoder @ {config_name_or_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(config_name_or_path)
        self.bert = AutoModel.from_pretrained(config_name_or_path).to(device)
        logger.info(f"Encoder loaded.")
        self.warmed: bool = False

    def embed_sentences(self, sentences: List[str]):
        if self.warmed:
            padding = True
        else:
            padding = "max_length"
            self.warmed = True
        batch = self.tokenizer.batch_encode_plus(
            sentences,
            return_tensors="pt",
            max_length=64,
            truncation=True,
            padding=padding
        )
        batch = {k: v.to(device) for k, v in batch.items()}

        fw = self.bert.forward(**batch)
        return fw.pooler_output

    def forward(self, sentences: List[str]):
        return self.embed_sentences(sentences)


def test():
    encoder = BERTEncoder("bert-base-cased")
    sentences = ["this is one", "why not another"]
    encoder.embed_sentences(sentences)
