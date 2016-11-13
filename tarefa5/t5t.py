import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import KFold
from sklearn.grid_search import GridSearchCV
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
import operator

n_cores = 2;
skf_nSplits = 5;
data = np.genfromtxt('train.csv', delimiter=',', dtype=str, usecols=range(1, 33));
values = np.loadtxt('train.csv', delimiter=',', usecols=(0,));
data_test_professor = np.genfromtxt('test.csv', delimiter=',', dtype=str, usecols=range(0, 32));
total_data = np.concatenate((data, data_test_professor), axis=0)

# dataList= data.tolist()
# dataList_test= data_test.tolist()
total_data_list = total_data.tolist()
indices = [0, 1, 2, 9, 12, 13, 17, 18, 20, 22, 23, 24, 25, 26, 30, 31]

for line in range(0, len(total_data_list)):
    for index in indices:
        total_data_list[line][index] = float(total_data_list[line][index])

data_frame = pd.DataFrame(total_data_list)
data_frame_converted = pd.get_dummies(data_frame)

X_total = np.array(data_frame_converted)
X_data_train = X_total[0:9000, :];
X_data_test_professor = X_total[9000:, :];



parameters_gbm = {'n_estimators': [30, 70, 100, 150, 200, 300, 500], 'loss':['lad'],
                  'max_depth': [3, 5],
                  'max_features': [5, 10, 15, 20, 25, None], 'warm_start':[True],  'learning_rate':[0.1, 0.05, 0.01]}



randomForestObj = RandomForestRegressor()
gbmObj = GradientBoostingRegressor()

skf = KFold(n_splits=skf_nSplits)
bestScore = 9999;

bestScore_rf = 9999;
bestScore_gbm = 9999;

rf_n_estimators_best = 0;
rf_max_features_best = 0;
rf_max_depth_best = 0;

gbm_n_estimators_best = 0;
gbm_max_features_best = 0;
gbm_max_depth_best = 0;
gbm_learning_rate_best = 0
gbm_warm_start_best = True
gbm_loss_best = 'lad'


svm_avarege_MAE = 0
rf_avarege_MAE = 0
gbm_avarege_MAE = 0
mlp_avarege_MAE = 0

# print "----------Avaliando qual regressor tem MAE medio melhor-----------"
# for train_index_skf, test_index_skf in skf.split(X_data_train):
#     X_train_skf, X_test_skf = \
#         X_data_train[train_index_skf], X_data_train[test_index_skf];
#
#     Y_train_skf, Y_test_skf = \
#         values[train_index_skf], values[test_index_skf];
#
#     # print "Fazendo o GridSearsh para a Random Forest regressor...."
#     # clf2 = GridSearchCV(randomForestObj, param_grid=parameters_rf, scoring='neg_mean_absolute_error')
#     # clf2.fit(X_train_skf, Y_train_skf)
#     # print "Finalizado o GridSearsh para a Random Forest regressor."
#     # print "MAE do Random Forest obtido para o conjunto de treino: " + str(
#     #     -clf2.best_score_) + "com os parametros: " + str(clf2.best_params_)
#     # rfPreditcTestKfold = clf2.predict(X_test_skf);
#     # MAE_RF = mean_absolute_error(Y_test_skf, rfPreditcTestKfold)
#     # print "MAE_RF obtido para o conjunto de teste: " + str(MAE_RF)
#     # rf_avarege_MAE = rf_avarege_MAE + MAE_RF
#
#     print "Fazendo o GridSearsh para o Gradien Boosting regressor...."
#     clf3 = GridSearchCV(gbmObj, param_grid=parameters_gbm, scoring='neg_mean_absolute_error')
#     clf3.fit(X_train_skf, Y_train_skf)
#     print "Finalizado o GridSearsh para a Gradien Boosting regressor."
#     print "MAE do Gradien Boosting obtido para o conjunto de treino: " + str(
#         -clf3.best_score_) + "com os parametros: " + str(clf3.best_params_)
#     gbmPreditcTestKfold = clf3.predict(X_test_skf);
#     MAE_GBM = mean_absolute_error(Y_test_skf, gbmPreditcTestKfold)
#     print "MAE_GBM obtido para o conjunto de teste: " + str(MAE_GBM)
#     gbm_avarege_MAE = gbm_avarege_MAE + MAE_GBM



MAE_mean_all = [svm_avarege_MAE, rf_avarege_MAE, gbm_avarege_MAE, mlp_avarege_MAE]
MAE_mean_all = [x / float(skf_nSplits) for x in MAE_mean_all]
print "MAE medio svm: " + str(MAE_mean_all[0])
print "MAE medio rf: " + str(MAE_mean_all[1])
print "MAE medio gbm: " + str(MAE_mean_all[2])
print "MAE medio mlp: " + str(MAE_mean_all[3])
names = ['svm', 'rf', 'gbm', 'mlp']
skf = KFold(n_splits=3)

print "\n----------Encontrando os melhores parametros----------"
for i in range(0, 1):
    #min_index, min_value = min(enumerate(MAE_mean_all), key=operator.itemgetter(1));
    #regressorAtual = names[min_index]

    #if (regressorAtual == 'gbm'):
    print "procurando os melhores parametros para GBM"
    for train_index_skf, test_index_skf in skf.split(X_data_train):
        X_train_skf, X_test_skf = \
            X_data_train[train_index_skf], X_data_train[test_index_skf];

        Y_train_skf, Y_test_skf = \
            values[train_index_skf], values[test_index_skf];
        print "Fazendo o GridSearsh para o Gradien Boosting regressor...."
        clf3 = GridSearchCV(gbmObj, param_grid=parameters_gbm, scoring='neg_mean_absolute_error', n_jobs=n_cores)
        clf3.fit(X_train_skf, Y_train_skf)
        print "Finalizado o GridSearsh para a Gradien Boosting regressor."
        print "MAE do Gradien Boosting obtido para o conjunto de treino: " + str(
            -clf3.best_score_) + "com os parametros: " + str(clf3.best_params_)
        gbmPreditcTestKfold = clf3.predict(X_test_skf);
        MAE_GBM = mean_absolute_error(Y_test_skf, gbmPreditcTestKfold)
        print "MAE_GBM obtido para o conjunto de teste: " + str(MAE_GBM)
        if (MAE_GBM < bestScore_gbm):
            bestScore_gbm = MAE_GBM
            gbm_n_estimators_best = clf3.best_params_['n_estimators']
            gbm_max_features_best = clf3.best_params_['max_features']
            gbm_max_depth_best = clf3.best_params_['max_depth']
            gbm_learning_rate_best = clf3.best_params_['learning_rate']
        print "melhor GBM parametros ate o momento: " + str(bestScore_gbm) + " n_estimators: " + str(
            gbm_n_estimators_best) \
              + " max_features: " + str(gbm_max_features_best) + " max_depth: " + str(gbm_max_depth_best) + \
              " learning_rate: " + str(gbm_learning_rate_best) + " loss: " + str(gbm_loss_best) + " warm_start: " + "True"

    print "\ncriando o regressor gbm com os melhores parametros"
    gbmRegression = GradientBoostingRegressor(n_estimators=gbm_n_estimators_best,
                                              max_features=gbm_max_features_best,
                                              max_depth=gbm_max_depth_best, learning_rate=gbm_learning_rate_best,
                                              warm_start=gbm_warm_start_best,loss=gbm_loss_best);
    gbmRegression.fit(X_data_train, values)
    print "computando a acuracia na base de treino....."
    dados_predicted = gbmRegression.predict(X_data_train)
    MAE_GBM = mean_absolute_error(values, dados_predicted)
    print "MAE obtido gbm foi de: " + str(MAE_GBM)
    print "aplicando o gbm nos dados de testes do professor...."
    dadosProfessor_predicted = gbmRegression.predict(X_data_test_professor);
    fname = "GBM" + str(i + 1)
    np.savetxt(fname=fname, delimiter=',', X=dadosProfessor_predicted, fmt='%f')
    print "-----------fim GBM---------------"

print "---------------------fim de execucao---------------------"


