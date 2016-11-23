# -*- coding: utf-8 -*-
"""
Instituto de Computacao - Universidade Estadual de Campinas

MO444/MC886 Aprendizado de maquina
2o semestre de 2016
Professor: Jacques Wainer

Aluno: Erick Ricardo Mattos - 139415

Exercicio 06
"""
# Para uso geral
import numpy as np
# Para ler o dataset
from sklearn.datasets import load_files
# Para fazer a cross validation
from sklearn.model_selection import StratifiedKFold
# Preprocessamento dos textos e criacao dos bags
import nltk
import string
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
# Para executar os classificadores
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC as SVM
from sklearn.ensemble import GradientBoostingClassifier as GBM
# Para calcular acuracia na previsao
from sklearn.metrics import accuracy_score as accuracy
# Para fazer PCA
from sklearn.decomposition import TruncatedSVD as PCA
# Procura de hiperparametros
from sklearn.model_selection import GridSearchCV

# Inputs do programa
pathContainerFolder = 'filesk'

# Carregando o dataset
try:
    dataset = load_files(pathContainerFolder, encoding='utf-8')
except Exception:
    print("An error happened.")
    exit(1)

X = np.array(dataset["data"])
y = np.array(dataset["target"])

#  Parte 1 - processamento de texto

# conversao de caracteres maiusculos para minusculos
X = list(map(lambda x: x.lower(), X))
# remoçao de pontuaçao
translator = str.maketrans({key: None for key in string.punctuation})
X = list(map(lambda x: x.translate(translator), X))
# remocao de stop words: CountVectorizer faz isso
# steming dos termos
stemmer = PorterStemmer()


def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed


def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems
# remocao dos termos que aparecem em um so documento
# min_df = float(2/len(dataset['filenames'])) -> faz a remocao
count_vect = CountVectorizer(min_df=float(2 / len(dataset['filenames'])),
                             tokenizer=tokenize, stop_words='english')
# bag of words no formato binario
X_count = count_vect.fit_transform(X)
X_binary = (X_count != 0).astype(int)
# bag of words no formato de term frequency
freq_vect = TfidfTransformer(use_idf=False)
X_tf = freq_vect.fit_transform(X_count)
print X_count.shape
#  Parte 2 - classificador multiclasse na matriz termo-documento original


