# Gender Data Substitution

__This repository aims at generating different versions of a given data on the gender level. Given a dataset, it can generate the following augmented datasets:__

1. original: represents the initial data before applying any augmentation technique.
2. gender-swapped: inverts all of the gendered language into the opposite gender, i.e. she ---> he.
3. balanced: combines both the original data with the gender-swapped one.
4. neutral: gender words are converted into gender free (neutral) words, i.e. she ---> they.
5. pro-stereotype: consists of the over represented stereotypes in the original data.
6. anti-stereotype: consists of the under represented stereotypes in the original data.

For more details on how to create these different versions of the dataset, refer to section 4 in the [report](../report.pdf).

## Setup environment

To install dependencies, run the following in the folder where [requirements.txt](requirements.txt) is stored:
```Bash
pip install -r requirements.txt
```
Then, download the model *en_core_web_sm* by typing:
```Bash
python -m spacy download en_core_web_sm
```



## Data Utilities
In the [data_utilities](./data_utilities) folder, you can find different json files that are used to apply gender data substitutions:
1. The [gender_names.json](./data/gender_names.json) consists of x male names and y female names mostly taken from .
2. The [gender_pairs.json](./data/gender_pairs.json) consisting of pairs of words, each pair has a female word with its corresponding male version. This data is taken from
3. The [neutral_pairs.json](./data/neutral_pairs.json) consists of pairs of words, each pair has a gender word with the corresponding gender free version of it.

## Datasets
As we are mainly concerned in gender data, we use both [CommonCrawl](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.646.4837&rep=rep1&type=pdf) and [WikipediaGenderEvents](https://github.com/PlusLabNLP/ee-wiki-bias/blob/master/data/final_manual.csv) corpora that can be found in [data_utilities](./data_utilities). Two Jupyter Notebooks are provided to generate the augmented versions of each corpus and will be released in [data](../data) to be used directly for the main project. They can be directly launched in Google Colab from here:

- <a href="https://colab.research.google.com/github/dinalzein/CSC/blob/main/data_substitution_CommonCrawl.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> for the CommonCrawl dataset.  

- <a href="https://colab.research.google.com/github/dinalzein/CSC/blob/main/data_substitution_WikipediaGenderEvents.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> for the Wikipedia dataset.


## Used materials and 3rd party code
Part of the code is taken from [Counterfactual Data Substitution](https://github.com/rowanhm/counterfactual-data-substitution) repository.
