import pandas as pd
import argparse
import json
from sklearn import metrics
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

    def __init__(self, learning_rate=1e-4, epochs=10):
        self.learning_rate = learning_rate
        self.epochs = epochs

    def read_csv(self, file):
        return pd.read_csv(file, sep='\t', header=None, names=self.names)

    def preprocess(self, df):
        df.label = df.label.apply(lambda label: int(label == 'positive'))
        for i, (slice, sentence) in enumerate(zip(df.slice, df.sentence)):
            start, end = map(int, slice.split(':'))
            df.at[i, 'sentence'] = f"{sentence[:start]}{self.start_of_term} {sentence[start:end]} {self.end_of_term}{sentence[end:]}"

    def train(self, trainfile):
        """Trains the classifier model on the training set stored in file trainfile"""
        df = self.read_csv(trainfile)
        self.preprocess(df)

        dataset = AbsaDataset(df)
        loader = DataLoader(dataset, batch_size=16, shuffle=True)

        print("Loading model and tokenizer...")
        self.model = DistilBertForSequenceClassification.from_pretrained(self.model_name, num_labels=dataset.num_categories)
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(self.model_name)
        print("Loaded")

        self.tokenizer.add_tokens([self.start_of_term, self.end_of_term])
        self.model.resize_token_embeddings(len(self.tokenizer))

        criterion = nn.BCEWithLogitsLoss()
        optimizer = Adam(self.model.parameters(), lr=self.learning_rate)

        for epoch in range(self.epochs):
            print(f"\n[Epoch {epoch+1}]")
            losses, y_true, y_pred = [], [], []
            for i, (texts, category_indices, labels) in enumerate(loader):
                labels = labels.type(torch.FloatTensor)
                inputs = self.tokenizer(list(texts), return_tensors='pt', padding=True)
                logits = self.model(**inputs)
                try:
                    logits = logits.logits
                except:
                    logits = logits[0]
                logits = logits[range(len(labels)), category_indices]

                y_true += labels.tolist()
                y_pred += (logits >= 0).type(torch.int8).tolist()

                loss = criterion(logits, labels)
                losses.append(loss.item())

                if (i+1) % 10 == 0:
                    print(f"  [{i-9:2}:{i:2}] {sum(losses)/len(losses)}")
                    losses = []
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(f"\n> Balanced accuracy: {metrics.balanced_accuracy_score(y_true, y_pred)}")
            print(f"\n> Accuracy: {metrics.accuracy_score(y_true, y_pred)}")
            print(f"> F1-score: {metrics.f1_score(y_true, y_pred)}\n")



    def predict(self, datafile):
        """Predicts class labels for the input instances in file 'datafile'
        Returns the list of predicted labels
        """
        predicted_labels = []

        df = self.read_csv(datafile)
        self.preprocess(df)

        dataset = AbsaDataset(df)
        loader = DataLoader(dataset, batch_size=16, shuffle=False)

        for texts, category_indices, _ in loader:
            labels = labels.type(torch.FloatTensor)
            inputs = self.tokenizer(list(texts), return_tensors='pt', padding=True)
            logits = self.model(**inputs).logits
            logits = logits[range(len(labels)), category_indices]
            predicted_labels += (logits >= 0).type(torch.int8).tolist()

        return predicted_labels



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epochs", type=int, default=10,
                        help="number of epochs")
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-4,
                        help="learning rate")

    args = parser.parse_args()
    print(f"> args:\n{json.dumps(vars(args), sort_keys=True, indent=4)}\n")
    trainfile = '../data/traindata.csv'

    classifier = Classifier(args.learning_rate, epochs=args.epochs)
    classifier.train(trainfile)


    
