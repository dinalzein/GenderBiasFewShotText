import logging
import random
import sys
import json
import spacy

class Substitutor:

    def __init__(self, gender_pairs, full_names, neutral_pairs, his_him=True, spacy_model='en_core_web_sm'):

        self.nlp = spacy.load(spacy_model)

        # This flag tells it whether or not to apply the special case intervention to him/his/her/hers
        self.his_him = his_him

        self.gender_pairs = TwoWayDict()
        for (male, female) in gender_pairs:
            self.gender_pairs[male.lower()] = female.lower()

        self.neutral_pairs = {}
        for (gender_word, neutral_word) in neutral_pairs:
            self.neutral_pairs[gender_word.lower()] = neutral_word.lower()


        self.full_names={}
        self.full_names['men']=[]
        self.full_names['women']=[]
        for male_name in full_names['men']:
            self.full_names['men'].append(male_name.lower())
        for female_name in full_names['women']:
            self.full_names['women'].append(female_name.lower())

        self.anonymizer = {}
        self.index=1


    def probablistic_substitute(self, input_texts):
        for text in input_texts:
            if bool(random.getrandbits(1)):
                yield self.invert_document(text)
            else:
                yield text

    def invert_text_gender(self, input_text):
        # Parse the doc
        doc = self.nlp(input_text)

        output = input_text

        # Walk through in reverse order making substitutions
        for word in reversed(doc):

            # Calculate inversion
            flipped = self.invert_word_gender(word)

            if flipped is not None:
                # Splice it into output
                start_index = word.idx
                end_index = start_index + len(word.text)
                output = output[:start_index] + flipped + output[end_index:]

        return output

    def invert_word_gender(self, spacy_word):

        flipped = None
        text = spacy_word.text.lower()
        # Handle base case
        if text in self.gender_pairs.keys():
            flipped = self.gender_pairs[text]

        # Handle name case

        elif text in self.full_names['men'] and spacy_word.ent_type_ == "PERSON":
            flipped = random.choice(self.full_names['women'])

        elif text in self.full_names['women'] and spacy_word.ent_type_ == "PERSON":
            flipped = random.choice(self.full_names['men'])


        # Handle special case (his/his/her/hers)
        elif self.his_him:
            pos = spacy_word.tag_
            if text == "him":
                flipped = "her"
            elif text == "his":
                if pos == "NNS":
                    flipped = "hers"
                else:  # PRP/PRP$
                    flipped = "her"
            elif text == "her":
                if pos == "PRP$":
                    flipped = "his"
                else:  # PRP
                    flipped = "him"
            elif text == "hers":
                flipped = "his"

        if flipped is not None:
            # Attempt to approximate case-matching
            return self.match_case(flipped, spacy_word.text)
        return None

    def invert_text_neutral(self, input_text):
        # Parse the doc
        doc = self.nlp(input_text)
        output = input_text

        # Walk through in reverse order making substitutions
        for word in reversed(doc):
            # Calculate inversion
            flipped = self.invert_word_neutral(word)

            if flipped is not None:
                # Splice it into output
                start_index = word.idx
                end_index = start_index + len(word.text)
                output = output[:start_index] + flipped + output[end_index:]

        return output

    def invert_word_neutral(self, spacy_word):

        flipped = None
        text = spacy_word.text.lower()
        # Handle base case
        if text in self.neutral_pairs.keys():
            flipped = self.neutral_pairs[text]

        # Handle name case

        elif spacy_word.ent_type_ == "PERSON":
            if text not in self.anonymizer.keys():
                self.anonymizer[text]="E"+str(self.index)
                self.index=self.index+1
            flipped = self.anonymizer[text]

        if flipped is not None:
            # Attempt to approximate case-matching
            return self.match_case(flipped, spacy_word.text)
        return None


    @staticmethod
    def match_case(input_string, target_string):
        # Matches the case of a target string to an input string
        # This is a very naive approach, but for most purposes it should be okay.
        if target_string.islower():
            return input_string.lower()
        elif target_string.isupper():
            return input_string.upper()
        elif target_string[0].isupper() and target_string[1:].islower():
            return input_string[0].upper() + input_string[1:].lower()
        else:
            return input_string




def load_json_pairs(input_file):
    with open(input_file, "r") as fp:
        pairs = json.load(fp)
    return pairs


class TwoWayDict(dict):
    def __setitem__(self, key, value):
        dict.__setitem__(self, key, value)
        dict.__setitem__(self, value, key)

    def __delitem__(self, key):
        dict.__delitem__(self, self[key])
        dict.__delitem__(self, key)

    def __len__(self):
        """Returns the number of connections"""
        return dict.__len__(self) // 2
