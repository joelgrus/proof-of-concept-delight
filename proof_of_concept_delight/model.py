#!/usr/bin/env python
# coding: utf8
"""Train a convolutional neural network text classifier on the
IMDB dataset, using the TextCategorizer component. The dataset will be loaded
automatically via Thinc's built-in dataset loader. The model is added to
spacy.pipeline, and predictions are available via `doc.cats`. For more details,
see the documentation:
* Training: https://spacy.io/usage/training

Compatible with: spaCy v2.0.0+
"""
from __future__ import unicode_literals, print_function, annotations

from typing import List
import random
from pathlib import Path
import random
import os
from collections import defaultdict
import json

import plac
import thinc.extra.datasets

import spacy
from spacy.util import minibatch, compounding

from proof_of_concept_delight.data import NewsArticle


class Model:
    def __init__(self,
                 labels: List[str],
                 model: str = None,
                 seed: int = 12) -> None:
        random.seed(seed)
        self.labels = labels

        if model is not None:
            self.nlp = spacy.load(model)  # load existing spaCy model
            print("Loaded model '%s'" % model)
        else:
            self.nlp = spacy.blank("en")  # create blank Language class
            print("Created blank 'en' model")

        # add the text classifier to the pipeline if it doesn't exist
        # nlp.create_pipe works for built-ins that are registered with spaCy
        if "textcat" not in self.nlp.pipe_names:
            textcat = self.nlp.create_pipe(
                "textcat", config={"exclusive_classes": True, "architecture": "simple_cnn"}
            )
            self.nlp.add_pipe(textcat, last=True)
        # otherwise, get it, so we can add labels to it
        else:
            textcat = self.nlp.get_pipe("textcat")

        # add label to text classifier
        for label in labels:
            textcat.add_label(label)

        self.labels = labels

    def _cats(self, article: NewsArticle):
        return {"cats": {label: article.site == label for label in self.labels}}

    def train(
        self,
        training_data: List[NewsArticle],
        validation_data: List[NewsArticle] = [],
        num_epochs: int = 100,
        batch_size: int = 20,
        shuffle: bool = True
    ) -> None:
        train_data = [(article.title, self._cats(article)) for article in training_data]
        valid_data = [(article.title, self._cats(article)['cats']) for article in validation_data]
        valid_texts, valid_annotations = zip(*valid_data)
        textcat = self.nlp.get_pipe("textcat")

        # get names of other pipes to disable them during training
        pipe_exceptions = ["textcat", "trf_wordpiecer", "trf_tok2vec"]
        other_pipes = [pipe for pipe in self.nlp.pipe_names if pipe not in pipe_exceptions]
        with self.nlp.disable_pipes(*other_pipes):  # only train textcat
            print("Training the model...")
            optimizer = self.nlp.begin_training()
            batch_sizes = compounding(4.0, 32.0, 1.001)

            for epoch in range(num_epochs):
                losses = {}
                if shuffle:
                    random.shuffle(train_data)

                batches = minibatch(train_data, size=batch_sizes)

                for batch in batches:
                    texts, annotations = zip(*batch)
                    self.nlp.update(texts, annotations, sgd=optimizer, drop=0.2, losses=losses)

                with textcat.model.use_params(optimizer.averages):
                    # evaluate on the dev data split off in load_data()
                    metrics = evaluate(self.nlp.tokenizer, textcat, valid_texts, valid_annotations)
                    print(epoch, losses["textcat"])
                    for label, scores in metrics.items():
                        print(
                            "{0}\t{1:.3f}\t{2:.3f}\t{3:.3f}".format(  # print a simple table
                            label,
                            scores["textcat_p"],
                            scores["textcat_r"],
                            scores["textcat_f"],
                            )
                        )

    def save(self, output_dir: str) -> None:
        self.nlp.to_disk(output_dir)
        with open(os.path.join(output_dir, 'labels.json'), 'w') as f:
            json.dump(self.labels, f)

    @staticmethod
    def load(output_dir: str) -> Model:
        with open(os.path.join(output_dir, 'labels.json')) as f:
            labels = json.load(f)
        model = Model(labels)
        model.nlp = spacy.load(output_dir)

        return model

    def predict(self, text: str):
        doc = self.nlp(text)
        return doc.cats

def evaluate(tokenizer, textcat, texts, cats):
    docs = (tokenizer(text) for text in texts)
    tp = 0.0  # True positives
    fp = 1e-8  # False positives
    fn = 1e-8  # False negatives
    tn = 0.0  # True negatives

    labels = sorted({key for cat in cats for key in cat})
    metrics = {label: defaultdict(int) for label in labels}

    for i, doc in enumerate(textcat.pipe(docs)):
        gold = cats[i]
        for label, score in doc.cats.items():
            if label not in gold:
                continue
            if label == "NEGATIVE":
                continue
            if score >= 0.5 and gold[label] >= 0.5:
                metrics[label]['tp'] += 1.0
            elif score >= 0.5 and gold[label] < 0.5:
                metrics[label]['fp'] += 1.0
            elif score < 0.5 and gold[label] < 0.5:
                metrics[label]['tn'] += 1
            elif score < 0.5 and gold[label] >= 0.5:
                metrics[label]['fn'] += 1
    for label in labels:
        tp = metrics[label]['tp']
        fp = metrics[label]['fp']
        tn = metrics[label]['tn']
        fn = metrics[label]['fn']

        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        if (precision + recall) == 0:
            f_score = 0.0
        else:
            f_score = 2 * (precision * recall) / (precision + recall)
        metrics[label].update({"textcat_p": precision, "textcat_r": recall, "textcat_f": f_score})
    return metrics