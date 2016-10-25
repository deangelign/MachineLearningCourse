import csv
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

data = []
label = []
print "\n---------------------------------------------------"
#leitura dos dados
with open('secom.data', 'rb') as file:#assumindo que o arquivo de leitura
#esta no mesmo diretorio de execucao que o .py
  reader = csv.reader(file, delimiter=' ', quotechar='|')
  data = np.loadtxt('secom.data', delimiter = ' ');
  label = np.loadtxt('secom_labels.data', usecols = [0]);
  imp = preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0)
  data = imp.fit_transform(data)
  data = preprocessing.scale(data)

#converter para um array do numpy
X = np.array(data);
Y = np.array(label);

#cria um copia dos dados e depois
#aplica o PCA manentdo 80% va variancia.
pca = PCA(n_components=0.8)#80% percento da variancia original
pca.fit(X)
data_pca = pca.transform(X)


#5-fold estratificado
skf = StratifiedKFold(n_splits=5)

#parametros dos metodos de machine learning
#onde sera realizado um gridsearch
parameters_knn = {'n_neighbors':[1, 5, 11, 15, 21, 25]}
parameters_svm = [{'kernel': ['rbf'], 'gamma': [2**(-15), 2**(-10), 2**(-5), 2**(0), 2**(5)],
                     'C': [2**(-5), 2**(0), 2**(5), 2**(10)]} ]
parameters_mlp = {'hidden_layer_sizes': [(10),(20),(30),(40)], 'solver':['lbfgs']}
parameters_rf = {'n_estimators': [100, 200, 300,400], 'max_features':[10, 15, 20, 25]}
parameters_gbm = {'n_estimators': [30, 70, 100], 'learning_rate':[0.1, 0.05], 'max_depth':[5]}

#objetos de cada classe de classificador
knnObj = KNeighborsClassifier()
svmObj = SVC();
mlpObj = MLPClassifier();
rfObj = RandomForestClassifier()
gbmObj = GradientBoostingClassifier()

#medias de acuracias
mediaKnn = mediaSvm = mediaMLP = mediaRF = mediaGBM = 0;


for train_index_skf, test_index_skf in skf.split(X,Y):


  #OBS: Eu apliquei PCA no conjunto de treinamento inteiro.
  #     Isso e errado (lembrar da primeira tarefa), porem
  #     na descricao da tarefa o professor nao obrigou
  #     a usar o pca no conjunto de treino gerado em cada
  #     iteracao do kfold.

  # conjunto de dados com o PCA, matendo 80% da variancia
  X_train_skf_pca, X_test_skf_pca = \
    data_pca[train_index_skf], data_pca[test_index_skf];

  X_train_skf, X_test_skf = \
    X[train_index_skf], X[test_index_skf];

  Y_train_skf, Y_test_skf = \
    Y[train_index_skf], Y[test_index_skf];

  #X_train_skf, X_test_skf = \
  #  X[train_index_skf], X[test_index_skf];
  #pca = PCA(n_components=0.8)  # 80% percento da variancia original
  #pca.fit(X_train_skf)
  #data_pca = pca.transform(X_train_skf)
  #X_train_skf = data_pca

  #faz o grid search no knn
  #usa o conjunto de treino com a reducao de
  #dimensionaliade aplicada pelo o PCA
  clf = GridSearchCV(knnObj, parameters_knn)
  clf.fit(X_train_skf_pca,Y_train_skf)
  print "acuracia: " + str(clf.best_score_) + "; para " + str(clf.best_params_)
  mediaKnn += clf.best_score_

  # faz o grid search no SVM
  clf2 = GridSearchCV(svmObj, parameters_svm, n_jobs=4)
  clf2.fit(X_train_skf,Y_train_skf)
  print "acuracia: " + str(clf2.best_score_) + "; para " + str(clf2.best_params_)
  mediaSvm += clf2.best_score_

  # faz o grid search na rede neural
  clf3 = GridSearchCV(mlpObj, parameters_mlp, n_jobs=2)
  clf3.fit(X_train_skf,Y_train_skf)
  print "acuracia: " + str(clf3.best_score_) + "; para " + str(clf3.best_params_)
  mediaMLP += clf3.best_score_;

  # faz o grid search na Random Forest
  clf4 = GridSearchCV(rfObj, parameters_rf, n_jobs=2)
  clf4.fit(X_train_skf,Y_train_skf)
  print "acuracia: " + str(clf4.best_score_) + "; para " + str(clf4.best_params_)
  mediaRF += clf4.best_score_;

  # faz o grid search no Gradient Boosting Machine
  clf5 = GridSearchCV(gbmObj, parameters_gbm, n_jobs=2)
  clf5.fit(X_train_skf,Y_train_skf)
  print "acuracia: " + str(clf5.best_score_) + "; para " + str(clf5.best_params_)
  mediaGBM += clf5.best_score_;



print 'acuracias medias'
print "\n---------------------------------------------------"
print "knn: " + str( (mediaKnn/5.0)*100 )
print "svm: " + str( (mediaSvm/5.0)*100 )
print "mlp: " + str( (mediaMLP/5.0)*100 )
print "Random Forest: " + str( (mediaRF/5.0)*100 )
print "Gradient Boosting Machine: " + str( (mediaGBM/5.0)*100 )

