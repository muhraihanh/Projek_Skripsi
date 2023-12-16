import pandas as pd

# Evaluation
class Evaluation:
    def __init__(self, extracted_keyphrase, golden_keyphrase):
        self.extracted_keyphrase = extracted_keyphrase
        self.golden_keyphrase =  golden_keyphrase[0].split(';')
        self.true_positives = 0
        self.false_positives = 0
        self.false_negatives = 0
        self.precision = 0
        self.recall = 0
        self.f1_score = 0

    def evaluate(self):
        self.true_positives = sum(word.lower() in map(str.lower, self.golden_keyphrase) for word in map(str.lower, self.extracted_keyphrase))
        self.false_positives = len(self.extracted_keyphrase) - self.true_positives
        self.false_negatives = len(self.golden_keyphrase) - self.true_positives

        self.precision = self.true_positives / (self.true_positives + self.false_positives) if (self.true_positives + self.false_positives) > 0 else 0
        self.recall = self.true_positives / (self.true_positives + self.false_negatives) if (self.true_positives + self.false_negatives) > 0 else 0
        self.f1_score = 2 * (self.precision * self.recall) / (self.precision + self.recall) if (self.precision + self.recall) > 0 else 0

    # def get_results(self):
    #     results = {
    #         'precision': self.precision,
    #         'recall': self.recall,
    #         'f1_score': self.f1_score
    #     }
    #     return results




