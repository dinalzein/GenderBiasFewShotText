# Gender Bias Few-shot Text
This repository is forked from [FewShotText](https://github.com/tdopierre/FewShotText) and is coming along with the paper cited in the [Citation](#citation) section. It aims at adding methods to mitigate gender bias in few-shot text models when dealing with gender based datasets.


## Introdcution
In few-shot classification for NLP tasks, we are interested in learning a classifier to recognize classes which were not seen at training time, provided only a handful of labeled examples in the inference regime. To do so, significant progress in few-shot classification has featured meta-learning, a parameterized model for learning algorithm, that is trained on episodes, each corresponding to a different classification mini-task -- each episodes comes with a small labeled support set and query set. Although these NLP models have shown success in classification tasks, they are based on a handful of labeled data so that they are prone to propagate all kind of bias found in language models. In this work we focus on gender bias found in text corpora. Few-shot NLP models use a pre-trained Transformer neural network to get an embedding representation of text samples that, unless addressed, are prone to learn intrinsic gender-bias in the dataset. We present different approaches that have shown success in quantifying and mitigating gender-bias in the transformer models. From these approaches, we propose different methods to quantify and mitigate bias in meta-learning for NLP tasks.

A [report](./report.pdf) has been made to explain in-details the work has been done and the few-shot models used to evaluate our strategies on.

## Datasets
We are using two different datasets [CommonCrawl](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.646.4837&rep=rep1&type=pdf) and [WikipediaGenderEvents](https://github.com/PlusLabNLP/ee-wiki-bias/blob/master/data/final_manual.csv). To get the augmented versions for each dataset, run the jupyter notebooks [data_substitution_CommonCrawl](./gender_data_substitution/src/data_substitution_CommonCrawl.ipynb) and [data_substitution_WikipediaGenderEvents](./gender_data_substitution/src/data_substitution_WikipediaGenderEvents.ipynb) in the directory stored in. After executing these notebooks, the different augmented versions of each dataset will be asserted in [data](./data). For more information on how these notebooks work, please refer to [README.md](./gender_data_substitution/README.md).


## Environment setup
```bash
# Create environment
python3 -m virtualenv .venv --python=python3.6

# Install environment
.venv/bin/pip install -r requirements.txt

# Activate environment
source .venv/bin/activate
```

## Fine-tuning BERT on the MLM task
To fine-tune BERT on the datasets in [Datasets](#datasets), run:
```Bash
utils/scripts/fine_tunning.sh
```
## Training a few-shot model
To run the the few-shot networks on the datasets in [Datasets](#datasets), simply run:
```Bash
utils/scripts/training_models.sh
```

### Citation
To cite this work in your publications:
```bash
@article{dopierre2021neural,
    title={A Neural Few-Shot Text Classification Reality Check},
    author={Dopierre, Thomas and Gravier, Christophe and Logerais, Wilfried},
    journal={arXiv preprint arXiv:2101.12073},
    year={2021}
}
```
