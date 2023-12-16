import re
from nltk.tag import CRFTagger as NLTKCRFTagger
from pathlib import Path
import os.path as path



# PraPemrosesan
class PraPemrosesan:
    def __init__(self, judul, abstrak):
        self.judul = judul
        self.abstrak = abstrak
        self.gabungan = ""
        self.token = []
        self.pos_tag = []
        self.noun_phrase = []
    
    def gabung_string(self):
        self.gabungan = self.judul + " " + self.abstrak
        # judul_abstrak.append(self.gabungan)

    def cleansing(self):
        # Strip leading and trailing spaces
        text = self.gabungan
        result = text.strip()

        # Store the last char
        last_char = result[-1]

        # Split by full-stop
        result = result.split('.')

        # And remove empty string from the list
        if result[-1] == '':
            del result[-1]

        # Add fult-stop to segments
        result = [(e + '.') for e in result]

        # Remove the extra full-stop by checking the last char
        if last_char != '.':
            result[-1] = result[-1][:-1]
        # class BasicSentenceSegmentator
        # sent_segm = BasicSentenceSegmentator()
        # segments = sent_segm.segmentate(self.gabungan)

        segments = result
        # class CustomTokenizer
        for i in range(len(segments)):

            # Transform to lower case
            segments[i] = segments[i].lower()

            # Replace new line and selecter characters with space
            segments[i] = re.sub(r'[=\n]', " ", segments[i])

            # Add spaces around to selected special marks
            segments[i] = re.sub(r'([/(),.?])', r" \1 ", segments[i])

            # Split by space
            segments[i] = segments[i].split(' ')

            # Remove empty string from the list
            segments[i] = [e for e in segments[i] if e != '']

        self.token = segments
        # tokenizer = CustomTokenizer()
        # self.token = tokenizer.tokenize(segments)
        # token_list.append(self.token)
    
    

    def pos_tagging(self):
        model_path = 'TaggerModel\post_tagger.model'
   
        self.pos_tag = self.tag(model_path, self.token)

    # extract()
    def extract(self):
     
        self.noun_phrase = self.select_term(self.pos_tag)


    def tag(self, model_path, sents):

        # Check the type of 'sent' argument
        if not isinstance(sents, list) or not isinstance(sents[0], list):
            raise TypeError(
                "invalid argument type of 'words', required : list of string list")

        # Merge sentence segments into a single sequence
        text = []
        for sent in sents:
            text += sent

        # Tagging process
        ct = NLTKCRFTagger()
        ct.set_model_file(model_path)
        pairs = ct.tag_sents([text])[0]
        # DEBUG

        # Resegmentate the sequence
        result = []
        temp = []
        for pair in pairs:
            temp.append(pair)
            if pair == ('.', 'Z'):
                result.append(temp)
                temp = []
        result.append(temp)

        return result

    def select_term(self, tagged_tokens):

        if len(tagged_tokens) > 1:
            ovrall_tokens = self._comb_sen(tagged_tokens)
        else:
            ovrall_tokens = tagged_tokens[0]

        # RULES FOR COMPOSITION :
        # 1. Multiple sequential adjectives and nouns are composed together if at least followed by a noun.
        # 2. Multiple sequential nouns with no adjectives leading are composed together.
        # 3. Other tags than noun are ending the composition chain.
        # 4. Single or multiple adjectives are not counted if not followed at least by a single noun.

        # VARIABLE :
        storage = []
        las_pair = (None, None)
        cur_com = []

        # ALGORITHM :
        # 1. Iterate on each of the splitted words.
        for cur_pair in ovrall_tokens:

            cur_word = cur_pair[0]
            cur_tag = cur_pair[1]
            las_tag = las_pair[1]

            # 2. Check the corresponding word tag :
            # If the corresponding tag is noun,
            if cur_tag in ['NN', 'NNP', 'NND', 'FW']:
                # And prior tag is noun or empty,
                if las_tag in ['NN', 'NNP', 'NND', 'FW'] or not cur_com:
                    # Add the word into composition.
                    cur_com.append(cur_pair)
                # Else the composed phrases is checked :
                else:
                    # If the composed phrases contains at least one noun,
                    if self._cont_noun(cur_com):
                        # Store to the storage.
                        storage.append(cur_com.copy())
                    # Drop the composed phrase.
                    cur_com.clear()
                    # Add the word into composition.
                    cur_com.append(cur_pair)

            # If the corresponding tag is adjective
            elif cur_tag in ['JJ']:
                # And prior tag is either noun or adjective or empty.
                if las_tag in ['JJ', 'NN', 'NNP', 'NND', 'FW'] or not cur_com:
                    # Add the word into composition.
                    cur_com.append(cur_pair)
                # Else the composed phrases is checked :
                else:
                    # If the composed phrases contains at least one noun,
                    if self._cont_noun(cur_com):
                        # Store to the storage.
                        storage.append(cur_com.copy())
                    # Drop the composed phrase.
                    cur_com.clear()
                    # Add the word into composition.
                    cur_com.append(cur_pair)

            # If the corresponding tag is neither noun or adjective, check :
            else:
                # If the composed phrases contains at least one noun,
                if self._cont_noun(cur_com):
                    # Store to the storage.
                    storage.append(cur_com.copy())
                # Drop the composed phrase.
                cur_com.clear()
                # If the words are special marks (e.g. : ".", ",", "?")
                if cur_word in ['.', '!', '?']:
                    # Store to the storage.
                    storage.append([cur_pair])

            # 3. Store the current pair as latest pair.
            las_pair = cur_pair
        # 4. After the iteration is finished, the composed phrases is checked:
        # If it contains at least one noun,
        if self._cont_noun(cur_com):
            # Store to the storage.
            storage.append(cur_com.copy())

        # 5. Return the combined pairs from storage
        return [self._pairs_to_phr(pair_list) for pair_list in storage]

    @staticmethod
    def _pairs_to_phr(pair_list):
        result = ''
        for pair in pair_list:
            result += (pair[0] + " ")
        return result.strip()

    @staticmethod
    def _comb_sen(tagged_sens):
        ovrall_tokens = []
        for sen in tagged_sens:
            # Skip if the segment contain no token
            if sen == []:
                continue
            ovrall_tokens += sen
        return ovrall_tokens

    @staticmethod
    def _cont_noun(tagged_words):
        # Check if a list of tagged words contains at least a noun.
        for pair in tagged_words:
            word_tag = pair[1]
            if word_tag in ['NN', 'NNP', 'NND', 'FW']:
                return True
        return False





