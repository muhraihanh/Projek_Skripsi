# =================================== EMBEDDING PROCESS =================================
import torch
from transformers import RobertaTokenizerFast, RobertaModel

# EmbeddingProcessor = Embedding
class Embedding():
    def __init__(self, noun_phrase="default", max_length=512):
        self.tokenizer = RobertaTokenizerFast.from_pretrained("cahya/roberta-base-indonesian-522M")
        self.model = RobertaModel.from_pretrained("cahya/roberta-base-indonesian-522M")
        self.noun_phrase = noun_phrase
        self.max_length = max_length
        self.hasil_embedding = []

    def process_embedding(self):
        # flat_noun_phrase = [
        #     word for sublist in self.noun_phrase for word in sublist
        # ]  # Flatten the list

        tokens = self.tokenizer(
            self.noun_phrase,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

        with torch.no_grad():
            outputs = self.model(**tokens)

        # Mendapatkan embedding dari output terakhir
        embeddings = outputs.last_hidden_state
        # Menghitung rata-rata embedding
        average_embedding = torch.mean(embeddings, dim=1)
        self.hasil_embedding = average_embedding
        # av_shape = average_embedding.shape
        # print("shape of embedding:", av_shape)
       