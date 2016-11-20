import numpy as np
from  sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
import string
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

random_state_seed = 0;
stemmer = PorterStemmer()
#remove stop words and stem the words
#------------------------------------------------------------------------
def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed


#before
def tokenizeB(text):
     text = "".join([ch for ch in text if ch not in string.punctuation])
     tokens = nltk.word_tokenize(text)
     stems = stem_tokens(tokens, stemmer)
     return stems

#after
def tokenizeA(text):
    tokens = nltk.word_tokenize(text)
    tokens = [i for i in tokens if i not in string.punctuation]
    stems = stem_tokens(tokens, stemmer)
    return stems
#-------------------------------------------------------------------------
dataset = load_files(container_path="filesk")
categorias_names = dataset.target_names;
categorias = dataset.target
#----------------------------------------------------------------------------------------------------------
#Por motivos de limitacao na memoria RAM sera primeiro feito todos os testes na matriz binaria e em seguida
#sera realizado os testes com a matriz de term frequency
#----------------------------------------------------------------------------------------------------------

vect_binary = CountVectorizer(tokenizer=tokenizeA, stop_words='english', lowercase=True,strip_accents='ascii',
                              min_df=2,binary=True, analyzer="word")
BAG_DATA_Binary = np.array(vect_binary.fit_transform(dataset.data).toarray())

X_TRAIN_BINARY, X_TEST_BINARY, Y_TRAIN_BINARY, Y_TEST_BINARY = train_test_split(BAG_DATA_Binary, categorias,
                                                                                test_size=0.2,
                                                                                random_state=random_state_seed);
del BAG_DATA_Binary
naiveBayesMNB_OBJ = MultinomialNB()
naiveBayesMNB_OBJ.fit(X_TRAIN_BINARY, Y_TRAIN_BINARY)
print("-------------Naive Bayes classifier for multinomial models with binary matrix--------")
print(metrics.classification_report(Y_TEST_BINARY, naiveBayesMNB_OBJ.predict(X_TEST_BINARY),
                                    target_names=categorias_names));
print("accuracy: {0}%".format(100*accuracy_score(Y_TEST_BINARY, naiveBayesMNB_OBJ.predict(X_TEST_BINARY))))
print
lrObj = LogisticRegression(C=10000)
lrObj.fit(X_TRAIN_BINARY, Y_TRAIN_BINARY)
print("-------------Logistic Regression with binary matrix--------")
print(metrics.classification_report(Y_TEST_BINARY, lrObj.predict(X_TEST_BINARY),target_names=categorias_names));
print("accuracy: {0}%".format(100*accuracy_score(Y_TEST_BINARY, lrObj.predict(X_TEST_BINARY))))
print
del X_TRAIN_BINARY
del X_TEST_BINARY
del Y_TRAIN_BINARY
del Y_TEST_BINARY

vect = CountVectorizer(tokenizer=tokenizeB, stop_words='english', lowercase=True,strip_accents='ascii', min_df=2,
                       analyzer="word")
TermFrequency = TfidfTransformer(use_idf=False)
BAG_DATA = np.array(vect.fit_transform(dataset.data).toarray())
TermFrequencyData =  np.array(TermFrequency.fit_transform(BAG_DATA).toarray())
X_TRAIN_TF, X_TEST_TF, Y_TRAIN_TF, Y_TEST_TF = train_test_split(TermFrequencyData, categorias, test_size=0.2,
                                                                random_state=random_state_seed);
del BAG_DATA
del TermFrequencyData
del dataset
del categorias
lrObj2 = LogisticRegression(C=10000)
lrObj2.fit(X_TRAIN_TF, Y_TRAIN_TF)
print("-------------Logistic Regression with term frequency--------")
print(metrics.classification_report(Y_TEST_TF, lrObj2.predict(X_TEST_TF),target_names=categorias_names));
#print("-------------------------------------------")
print("accuracy: {0}%".format(100*accuracy_score(Y_TEST_TF, lrObj2.predict(X_TEST_TF))))
print

pca = PCA(n_components=0.99,copy=False)
pca.fit(X_TRAIN_TF)
X_TRAIN_TF_PCA = pca.transform(X_TRAIN_TF)
X_TEST_TF_PCA = pca.transform(X_TEST_TF)
del X_TRAIN_TF
del X_TEST_TF

svmObj = SVC(kernel='rbf')
svmObj.fit(X_TRAIN_TF_PCA, Y_TRAIN_TF)
print("-------------SVM classifier with term frequency and PCA--------")
print(metrics.classification_report(Y_TEST_TF, svmObj.predict(X_TEST_TF_PCA),target_names=categorias_names));
print("accuracy: {0}%".format(100*accuracy_score(Y_TEST_TF, svmObj.predict(X_TEST_TF_PCA))))
print

gbmObj = GradientBoostingClassifier(warm_start=True)
gbmObj.fit(X_TRAIN_TF_PCA, Y_TRAIN_TF)
print("----------Gradient Boosting classifier with term frequency and PCA--------")
print(metrics.classification_report(Y_TEST_TF, gbmObj.predict(X_TEST_TF_PCA),target_names=categorias_names));
print("accuracy: {0}%".format(100*accuracy_score(Y_TEST_TF, gbmObj.predict(X_TEST_TF_PCA))))
print

rfObj = RandomForestClassifier(warm_start=True, n_estimators=1000)
rfObj.fit(X_TRAIN_TF_PCA, Y_TRAIN_TF)
print("----------Random Forest classifier with term frequency and PCA--------")
print(metrics.classification_report(Y_TEST_TF, rfObj.predict(X_TEST_TF_PCA),target_names=categorias_names));
print("accuracy: {0}%".format(100*accuracy_score(Y_TEST_TF, rfObj.predict(X_TEST_TF_PCA))))
print

del X_TRAIN_TF_PCA
del X_TEST_TF_PCA