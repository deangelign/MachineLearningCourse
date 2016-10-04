import csv
import numpy as np
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import KFold
from sklearn.svm import SVC

data = []
label = []
print "\n---------------------------------------------------"
#leitura dos dados
with open('data.csv', 'rb') as file:#assumindo que o arquivo de leitura
#esta no mesmo diretorio de execucao que o .py
  reader = csv.reader(file, delimiter=' ', quotechar='|')
  for row in reader:
    data.append(eval(str(row).replace("'", ''))[0:166])
    label.append(eval(str(row).replace("'", ''))[-1])
data.pop(0)
label.pop(0)

#converter para um array do numpy
X = np.array(data);
Y = np.array(label);

#5-fold estratificado
skf = StratifiedKFold(label,n_folds=5)

#variacoes dos valores de gamma e c para o grid search
c_variable = [2**-5, 2**-2, 2**0, 2**2, 2**5]
gamma_variable = [2**-15, 2**-10, 2**-5, 2**0,  2**5 ]

#variaveis auxiliares
numeroIteracoes = 0;
score_medio = 0;

#valicada externa
for train_index_skf, test_index_skf in skf:
  X_train_skf, X_test_skf = \
    X[train_index_skf], X[test_index_skf];

  Y_train_skf, Y_test_skf = \
    Y[train_index_skf], Y[test_index_skf];

  #conjunto de treino da validacao externa e feito um 3-fold
  kf = KFold(n = len(X_train_skf) , n_folds=3)

  # outras variaveis auxiliares
  c_best = 0;
  gamma_best = 0;
  score_best = 0;

  #avalia todas as configuracoes internas do 3-fold
  for train_index_kf, test_index_kf in kf:
    X_train_kf, X_test_kf = \
      X_train_skf[train_index_kf], X_train_skf[test_index_kf]

    Y_train_kf, Y_test_kf = \
      Y_train_skf[train_index_kf], Y_train_skf[test_index_kf]

    #grid search
    for i in c_variable:
      for j in gamma_variable:
        #cria um classifiador SVM com kernel do tipo
        # rbf para a validacao interna
        clf = SVC(C=i,kernel='rbf',gamma=j)
        clf.fit(X_train_kf, Y_train_kf)
        score = clf.score(X_test_kf,Y_test_kf)
        # para essa configuracao interna, esta e a melhor
        # combinacao de parametros?
        if(score > score_best):
          score_best = score;
          c_best = i;
          gamma_best = j;
  #cria um classifiador SVM com kernel do tipo rbf
  # para a validacao externa
  clf_externo = SVC(C=c_best, kernel='rbf', gamma=gamma_best)
  clf_externo.fit(X_train_skf, Y_train_skf)
  score_atual = clf_externo.score(X_test_skf, Y_test_skf)
  score_medio +=  score_atual
  numeroIteracoes +=   1;
  print "Melhores valores para a configucao interna atual: c = "\
        + str(c_best) + " gamma = " + str(gamma_best)

  print "Acuracia na validacao de fora: " + str(score_atual*100) + "%"
  print

#computa a acuracia media
score_medio = score_medio/numeroIteracoes
print "Acuracia media: " + str(score_medio*100) + "%"


#faz o 3-fold no conjunto _todo para encontrar
# os parametros c e gamma que serao usados na versao final
kf = KFold(n = len(X) , n_folds=3)
c_best = 0;
gamma_best = 0;
score_best = 0;
for train_index, test_index in kf:
  X_train, X_test = X[train_index], X[test_index];
  Y_train, Y_test = Y[train_index], Y[test_index];
  # grid search
  for i in c_variable:
    for j in gamma_variable:
      clf = SVC(C=i,kernel='rbf',gamma=j)
      clf.fit(X_train, Y_train)
      score = clf.score(X_test, Y_test)
      if (score > score_best):
          score_best = score;
          c_best = i;
          gamma_best = j;

print "---------------------------------------------------"
print "Melhores parametros  a serem usados no classificador final: c = "\
      + str(c_best) + " gamma = " + str(gamma_best) + \
      ". Obtendo uma acuracia de: " + str(score_best*100) + "%"
print "---------------------------------------------------"