import random
import os

from proof_of_concept_delight.data import load_articles
from proof_of_concept_delight.model import Model

OUTPUT_DIR = 'saved_model'

if __name__ == "__main__":
    random.seed(12)
    articles = load_articles('NewsArticles.csv')
    random.shuffle(articles)
    train_data = articles[:2000]
    validation_data = articles[2000:3000]
    test_data = articles[3000:]

    labels = sorted({article.site for article in articles})

    model = Model(labels, 'en_core_web_lg')

    model.train(train_data, validation_data, num_epochs=20)
    model.save(OUTPUT_DIR)

