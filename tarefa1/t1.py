import csv
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
#from sklearn.lda import LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

data = []
label = []

#leitura dos dados
with open('data1.csv', 'rb') as file:#assumindo que o arquivo de leitura 
#esta no mesmo diretorio de execucao que o .py
  reader = csv.reader(file, delimiter=' ', quotechar='|')
  for row in reader:
    data.append(eval(str(row).replace("'", ''))[0:166])
    label.append(eval(str(row).replace("'", ''))[-1])
data.pop(0)
label.pop(0)


X = np.array(data)
nSamples = len(X)
pca = PCA(n_components=0.8)#80% percento da variancia original
pca.fit(X)#gera a matriz de transformacao entre os espacos
data_pca = pca.transform(X)#aplica a transformacao nos dados originais e 
#matem 80% da variancia original
n_dimensions = len(data_pca[0])#numero de dimensoes necessarias para 
#manter os 80%
print "\nO numero de dimensoes minimo para manter 80% da variancia" + \
"dos dados originais e de:" + str(n_dimensions);

#criacao e predicoes dos classificadores de regressao linear 
#(com e sem PCA)
#Com PCA
lr_pca = LogisticRegression()
lr_pca.fit(data_pca[0:200], label[0:200])#treinando com as primeiras 200 amostras
lr_prediction_pca = lr_pca.predict(data_pca[200:nSamples])#testando com as demais amostras (276)
#sem PCA
lr_npca = LogisticRegression()
lr_npca.fit(data[0:200], label[0:200])
lr_prediction_npca = lr_npca.predict(data[200:nSamples])

#criacao e predicoes dos classificadores LDA (com e sem PCA)
#Com PCA
lda_pca = LinearDiscriminantAnalysis()
lda_pca.fit(data_pca[0:200], label[0:200])
lda_prediction_pca = lda_pca.predict(data_pca[200:nSamples])
#sem PCA
lda_npca = LinearDiscriminantAnalysis()
lda_npca.fit(data[0:200], label[0:200])
lda_prediction_npca = lda_npca.predict(data[200:nSamples])


#medicao da tabela de confusao
label_test = label[200:nSamples]
TP_lr_pca = TN_lr_pca = FP_lr_pca = FN_lr_pca = 0
TP_lr_npca = TN_lr_npca = FP_lr_npca = FN_lr_npca = 0
TP_lda_pca = TN_lda_pca = FP_lda_pca = FN_lda_pca = 0
TP_lda_npca = TN_lda_npca = FP_lda_npca = FN_lda_npca = 0
n_testSamples = len(label_test)

#0 - negative
#1 - positive
for i in range(0, 276):

  #lr pca
  if label_test[i] == lr_prediction_pca[i]:
    if label_test[i] == 1:
      TP_lr_pca += 1;
    else:
      TN_lr_pca += 1;

  if label_test[i] != lr_prediction_pca[i]:
    if label_test[i] == 1:
      FN_lr_pca += 1;
    else:
      FP_lr_pca += 1;

  # lr sem pca
  if label_test[i] == lr_prediction_npca[i]:
    if label_test[i] == 1:
      TP_lr_npca += 1;
    else:
      TN_lr_npca += 1;

  if label_test[i] != lr_prediction_npca[i]:
    if label_test[i] == 1:
      FN_lr_npca += 1;
    else:
      FP_lr_npca += 1;

  #lda pca
  if label_test[i] == lda_prediction_pca[i]:
    if label_test[i] == 1:
      TP_lda_pca += 1;
    else:
      TN_lda_pca += 1;

  if label_test[i] != lda_prediction_pca[i]:
    if label_test[i] == 1:
      FN_lda_pca += 1;
    else:
      FP_lda_pca += 1;

  # lda sem pca
  if label_test[i] == lda_prediction_npca[i]:
    if label_test[i] == 1:
      TP_lda_npca += 1;
    else:
      TN_lda_npca += 1;

  if label_test[i] != lda_prediction_npca[i]:
    if label_test[i] == 1:
      FN_lda_npca += 1;
    else:
      FP_lda_npca += 1;



#computa a acuracia em percetagem
print "\n"
print "PCA + Regressao logistica:"
print " " * 3 + "P\t N"  ;
print " " + "-" * 16;
print "P|" + str(TP_lr_pca) + "\t|" + str(FN_lr_pca) + "\t|";
print "N|" + str(FP_lr_pca) + "\t|" + str(TN_lr_pca) + "\t|";
print " " + "-" * 16;
acc = (100.0*(TP_lr_pca+TN_lr_pca))/n_testSamples;
print "acuracia (taxa de acerto): " + str(acc)
print "\n"

print "Regressao logistica:"
print " " * 3 + "P\t N"  ;
print " " + "-" * 16;
print "P|" + str(TP_lr_npca) + "\t|" + str(FN_lr_npca) + "\t|";
print "N|" + str(FP_lr_npca) + "\t|" + str(TN_lr_npca) + "\t|";
print " " + "-" * 16;
acc = (100.0*(TP_lr_npca+TN_lr_npca))/n_testSamples;
print "acuracia (taxa de acerto): " + str(acc)
print "\n"

print "PCA + LDA:"
print " " * 3 + "P\t N"  ;
print " " + "-" * 16;
print "P|" + str(TP_lda_pca) + "\t|" + str(FN_lda_pca) + "\t|";
print "N|" + str(FP_lda_pca) + "\t|" + str(TN_lda_pca) + "\t|";
print " " + "-" * 16;
acc = (100.0*(TP_lda_pca+TN_lda_pca))/n_testSamples;
print "acuracia (taxa de acerto): " + str(acc)
print "\n"

print "LDA:"
print " " * 3 + "P\t N"  ;
print " " + "-" * 16;
print "P|" + str(TP_lda_npca) + "\t|" + str(FN_lda_npca) + "\t|";
print "N|" + str(FP_lda_npca) + "\t|" + str(TN_lda_npca) + "\t|";
print " " + "-" * 16;
acc = (100.0*(TP_lda_npca+TN_lda_npca))/n_testSamples;
print "acuracia (taxa de acerto): " + str(acc)
print "\n"