#!/usr/bin/env python
# coding: utf-8

# Based on
# https://colab.research.google.com/github/allenai/longformer/blob/master/scripts/convert_model_to_long.ipynb
import torch
from transformers import BertForMaskedLM, BertTokenizerFast, LongformerForSequenceClassification, LongformerTokenizerFast, AutoTokenizer, RobertaTokenizer, BigBirdTokenizer, BigBirdForSequenceClassification, AutoModelForSequenceClassification
#from datasets import load_dataset
from transformers import Trainer, TrainingArguments
import csv
from sklearn.metrics import classification_report


try:
    from transformers.modeling_longformer import LongformerSelfAttention
except ImportError:
    from transformers import LongformerSelfAttention


# ### BertLong
#
# `BertLong` represents the "long" version of the `BERT` model. It replaces
# `BertSelfAttention` with `BertLongSelfAttention`, which is a thin wrapper
# around `LongformerSelfAttention`.


            
            
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor([self.labels[idx]])
        return item

    def __len__(self):
        return len(self.labels)



# Choose GPU
#import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def loadmodel(model_path, tokenizer_path):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)
    return tokenizer, model

    
def train_test():
    X_train = []
    Y_train = []
    
    with open('/home/s3494888/longformer/train.csv', encoding='utf-8') as train:
        data = csv.reader(train)
        for line in data:
            X_train.append(line[0])
            if line[1] == 'neg':
                Y_train.append(0)
            if line[1] == 'pos':
                Y_train.append(1)    
          
            
    X_test = []
    Y_test = []
    
    with open('/home/s3494888/longformer/test.csv', encoding='utf-8') as test:
        data = csv.reader(test)
        for line in data:
            X_test.append(line[0])
            if line[1] == 'neg':
                Y_test.append(0)
            if line[1] == 'pos':
                Y_test.append(1) 
            
    return X_train, Y_train, X_test, Y_test



def get_prediction(text, max_length, tokenizer, model):
    # prepare our text into tokenized sequence
    inputs = tokenizer(text, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to('cuda')
    # perform inference to our model
    outputs = model(**inputs)
    # get output probabilities by doing softmax
    probs = outputs[0].softmax(1)
    # executing argmax function to get the candidate label
    return probs.argmax()



def main():

    #model_path = '/data/s3494888/bertjelong/longformer/bert-base-dutch-4096'
    #model_path = '/data/s3494888/robbert-long/robbert-v2-dutch-base-4096'
    #model_path = 'GroNLP/bert-base-dutch-cased'
    #model_path = 'pdelobelle/robbert-v2-dutch-base'
    #model_path = 'markussagen/xlm-roberta-longformer-base-4096'
    model_path = 'flax-community/pino-bigbird-roberta-base'


    #tokenizer_path = 'GroNLP/bert-base-dutch-cased'
    #tokenizer_path = 'pdelobelle/robbert-v2-dutch-base'
    #tokenizer_path = 'xlm-roberta-base'
    #tokenizer_path = 'allenai/longformer-base-4096'
    tokenizer_path = 'flax-community/pino-bigbird-roberta-base'



    tokenizer, model = loadmodel(model_path, tokenizer_path)
    
    max_length = 2048

    X_train, Y_train, X_test, Y_test = train_test()

    
    tokens_train = tokenizer(X_train, padding=True, truncation=True, max_length=max_length)
    train_dataset = Dataset(tokens_train, Y_train) 

    training_args = TrainingArguments(
        output_dir='/data/s3494888/results',          # output directory
        num_train_epochs=3,              # total number of training epochs
        per_device_train_batch_size=2,  # batch size per device during training
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
        logging_steps=10,
        learning_rate=5e-03
    )

    trainer = Trainer(
        model=model,                         # the instantiated Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset
    )

    trainer.train()

    preds = []
    for x in X_test:
        preds.append(get_prediction(x, max_length, tokenizer, model).tolist())
    print(classification_report(Y_test, preds, zero_division=True, digits=3))
	


if __name__ == '__main__':
	main()
