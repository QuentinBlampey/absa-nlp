import pandas as pd
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
from torch import nn
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

from dataset import AbsaDataset

class Classifier:
    names = ['label', 'category', 'term', 'slice', 'sentence']
    start_of_term = '<sot>'
    end_of_term = '<eot>'
    model_name = 'distilbert-base-uncased'

    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def read_csv(self, file):
        return pd.read_csv(file, sep='\t', header=None, names=self.names)

    def preprocess(self, df):
        df.label = df.label.apply(lambda label: int(label == 'positive'))
        for i, (slice, sentence) in enumerate(zip(df.slice, df.sentence)):
            start, end = map(int, slice.split(':'))
            df.at[i, 'sentence'] = f"{sentence[:start]}{self.start_of_term} {sentence[start:end]} {self.end_of_term}{sentence[end:]}"

    def train(self, trainfile, epochs=2):
        """Trains the classifier model on the training set stored in file trainfile"""
        df = self.read_csv(trainfile)
        self.preprocess(df)

        dataset = AbsaDataset(df)
        loader = DataLoader(dataset, batch_size=16, shuffle=True)

        print("Loading model and tokenizer...")
        model = DistilBertForSequenceClassification.from_pretrained(self.model_name, num_labels=dataset.num_categories)
        tokenizer = DistilBertTokenizerFast.from_pretrained(self.model_name)
        print("Loaded")

        tokenizer.add_tokens([self.start_of_term, self.end_of_term])
        model.resize_token_embeddings(len(tokenizer))

        criterion = nn.BCEWithLogitsLoss()
        optimizer = Adam(model.parameters(), lr=self.learning_rate)

        for epoch in range(epochs):
            print(f"\n[Epoch {epoch}]")
            for texts, category_indices, labels in loader:
                labels = labels.type(torch.FloatTensor)
                inputs = tokenizer(list(texts), return_tensors='pt', padding=True)
                logits = model(**inputs).logits
                logits = logits[range(len(labels)), category_indices]

                loss = criterion(logits, labels)
                print(loss.item())
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


    def predict(self, datafile):
        """Predicts class labels for the input instances in file 'datafile'
        Returns the list of predicted labels
        """



if __name__ == '__main__':
    trainfile = '../data/traindata.csv'

    classifier = Classifier(1e-4)
    classifier.train(trainfile)


    
