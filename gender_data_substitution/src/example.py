import sys
from substitutor import Substitutor
from utils import load_json_pairs
import json
'''
this class presents an example on how to flip a sentence into the opposite gender
'''
# Example text which requires NER and POS information to properly invert
text = "Mike is nice, Lynn and James are nice"
data_path = '../data_utilities/'
gender_pairs = load_json_pairs(f'{data_path}/gender_pairs.json')
neutral_pairs = load_json_pairs(f'{data_path}/neutral_pairs.json')
full_names = json.loads(open(f'{data_path}/gender_names.json', "rb").readlines()[0])

# Initialise a substitutor with a list of pairs of gendered words (and optionally names)
substitutor = Substitutor(gender_pairs, full_names, neutral_pairs)

gender_flipped = substitutor.invert_text_gender(text)

neutral_flipped = substitutor.invert_text_neutral(text)

print(f"original text: {text}")
print(f"gender flipped text: {gender_flipped}")
print(f"gender free: {neutral_flipped}")
