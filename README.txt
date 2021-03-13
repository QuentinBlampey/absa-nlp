1. Name of the students
==========================================
Clément Piat
Quentin Blampey

2. Model Description
==========================================
We defined a very simple yet efficient model.

We decided to use a pretrained Transformer, as they perform
very well on most of the NLP tasks. DistilBert is a small, fast,
cheap and light version of BERT, thus very efficient for such
a small dataset. This model was loaded from The Hugging Face,
and we have chosen the DistilBertForSequenceClassification
model, as we want to produce a prediction for each of the 12
categories.

Thus, for a given text, we pass it into DistilBert, and get
the prediction associated to the corresponding category.
We are using the overused BCEWithLogits Loss from PyTorch.

We decided to add special tokens around the target terms so
that DistilBert knows the words it has to take care of.
E.g. "the free <sot> appetizer of olives <eot> was 
disappointing", where <sot> means "start of term" and <eot>
means "end of term". It's just like highlighting the target
terms to help DistilBert.

3. Dev Metrics
==========================================
> Accuracy: 0.999
> Balanced accuracy: 0.998
> F1-score: 0.999