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
data = np.genfromtxt('train.csv', delimiter = ',', dtype=str, usecols=range(1,33));
values = np.loadtxt('train.csv', delimiter = ',', usecols=(0,));
data_test_professor = np.genfromtxt('test.csv', delimiter = ',', dtype=str, usecols=range(0,32));
total_data = np.concatenate((data,data_test_professor),axis=0)

#dataList= data.tolist()
#dataList_test= data_test.tolist()
total_data_list = total_data.tolist()
indices = [0,1,2,9,12,13,17,18,20,22,23,24,25,26,30,31]


for line in range(0,len(total_data_list)):
    for index in indices:
        total_data_list[line][index] = float(total_data_list[line][index])

data_frame = pd.DataFrame(total_data_list)
data_frame_converted = pd.get_dummies(data_frame)

X_total=np.array(data_frame_converted)
X_data_train = X_total[0:9000,:];
X_data_test_professor = X_total[9000:,:];

parameters_svm = [{'kernel': ['rbf'], 'gamma': [2**(-15), 2**(-10), 2**(-5), 2**(0), 2**(5), 2**(10), 2**(15)],
                   'C': [2**(-10), 2**(-5), 2**(0), 2**(5), 2**(10)]} ]
parameters_rf = {'n_estimators': [50, 100, 200, 300,400, 500, 1000], 'max_features':[5, 10, 15, 20, 25], 'max_depth':[5,None]}
parameters_gbm = {'n_estimators': [10, 30, 70, 100, 150, 200, 300], 'learning_rate':[0.5, 0.1, 0.05, 0.01], 'max_depth':[3,5,10],
                  'max_features': [5, 10, 15, 20, 25,None]}
parameters_mlp = {'hidden_layer_sizes': [(10),(20),(30),(40)], 'solver':['lbfgs'],'activation':['logistic','relu'],
                  'learning_rate' : ['adaptive'], 'learning_rate_init':[0.001,0.01,0.1,1]}

svmObj = SVR();
randomForestObj = RandomForestRegressor()
gbmObj = GradientBoostingRegressor()
mlpObj = MLPRegressor()

skf = KFold(n_splits=skf_nSplits)
bestScore = 9999;

bestScore_svm = 9999;
bestScore_rf = 9999;
bestScore_gbm = 9999;
bestScore_mlp = 9999;

svm_c_best = 0;
svm_gamma_best = 0;
svm_kernel_best = "";

rf_n_estimators_best = 0;
rf_max_features_best = 0;
rf_max_depth_best = 0;

gbm_n_estimators_best = 0;
gbm_max_features_best = 0;
gbm_max_depth_best = 0;
gbm_learning_rate_best = 0;
gbm_warm_start_best = True
gbm_loss_best = 'lad'

mlp_hidden_layer_sizes_best = 0;
mlp_solver_best = ''
mlp_activation_best = ''
mlp_learning_rate_best = ''
mlp_learning_rate_init_best = 0;

svm_avarege_MAE = 0
rf_avarege_MAE = 0
gbm_avarege_MAE = 0
mlp_avarege_MAE = 0

iter = 0

print "----------Avaliando qual regressor tem MAE medio melhor-----------"
for train_index_skf, test_index_skf in skf.split(X_data_train):
    print "--------------iteracao " + str(iter) + "------------------"
    X_train_skf, X_test_skf = \
        X_data_train[train_index_skf], X_data_train[test_index_skf];

    Y_train_skf, Y_test_skf = \
        values[train_index_skf], values[test_index_skf];
    print
    print "Fazendo o GridSearsh para o SVM regressor...."
    clf = GridSearchCV(svmObj, param_grid=parameters_svm, scoring='neg_mean_absolute_error')
    clf.fit(X_train_skf, Y_train_skf)
    print "Finalizado o GridSearsh para o SVM regressor."
    print "MAE do SVM obtido para o conjunto de treino: " + str(-clf.best_score_) + "com os parametros: " + str(clf.best_params_)
    svmPreditcTestKfold = clf.predict(X_test_skf);
    MAE_SVM = mean_absolute_error(Y_test_skf,svmPreditcTestKfold)
    print "MAE_SVM obtido para o conjunto de teste: " + str(MAE_SVM)
    svm_avarege_MAE = svm_avarege_MAE + MAE_SVM
    print
    print "Fazendo o GridSearsh para a Random Forest regressor...."
    clf2 = GridSearchCV(randomForestObj, param_grid=parameters_rf, scoring='neg_mean_absolute_error')
    clf2.fit(X_train_skf, Y_train_skf)
    print "Finalizado o GridSearsh para a Random Forest regressor."
    print "MAE do Random Forest obtido para o conjunto de treino: " + str(
        -clf2.best_score_) + "com os parametros: " + str(clf2.best_params_)
    rfPreditcTestKfold = clf2.predict(X_test_skf);
    MAE_RF = mean_absolute_error(Y_test_skf, rfPreditcTestKfold)
    print "MAE_RF obtido para o conjunto de teste: " + str(MAE_RF)
    rf_avarege_MAE = rf_avarege_MAE + MAE_RF
    print
    print "Fazendo o GridSearsh para o Gradient Boosting regressor...."
    clf3 = GridSearchCV(gbmObj, param_grid=parameters_gbm, scoring='neg_mean_absolute_error')
    clf3.fit(X_train_skf, Y_train_skf)
    print "Finalizado o GridSearsh para a Gradient Boosting regressor."
    print "MAE do Gradien Boosting obtido para o conjunto de treino: " + str(
        -clf3.best_score_) + "com os parametros: " + str(clf3.best_params_)
    gbmPreditcTestKfold = clf3.predict(X_test_skf);
    MAE_GBM = mean_absolute_error(Y_test_skf, gbmPreditcTestKfold)
    print "MAE_GBM obtido para o conjunto de teste: " + str(MAE_GBM)
    gbm_avarege_MAE = gbm_avarege_MAE + MAE_GBM
    print
    print "Fazendo o GridSearsh para a MLP regressor...."
    clf4 = GridSearchCV(mlpObj, param_grid=parameters_mlp,  scoring='neg_mean_absolute_error')
    clf4.fit(X_train_skf, Y_train_skf)
    print "Finalizado o GridSearsh para a MLP regressor."
    print "MAE da MLP  obtido para o conjunto de treino: " + str(
        -clf4.best_score_) + " com os parametros: " + str(clf4.best_params_)
    mlpPreditcTestKfold = clf4.predict(X_test_skf);
    MAE_MLP = mean_absolute_error(Y_test_skf, mlpPreditcTestKfold)
    print "MAE_MLP obtido para o conjunto de teste: " + str(MAE_MLP)
    mlp_avarege_MAE = mlp_avarege_MAE + MAE_MLP
    iter = iter + 1
    print

print "MAE's Medios:"
MAE_mean_all = [svm_avarege_MAE,rf_avarege_MAE ,gbm_avarege_MAE, mlp_avarege_MAE]
MAE_mean_all = [x/float(skf_nSplits) for x in MAE_mean_all]
print "MAE medio svm: " + str(MAE_mean_all[0])
print "MAE medio rf: " + str(MAE_mean_all[1])
print "MAE medio gbm: " + str(MAE_mean_all[2])
print "MAE medio mlp: " + str(MAE_mean_all[3])
names = ['svm','rf','gbm','mlp']
skf = KFold(n_splits=3)

print "\n----------Encontrando os melhores parametros----------"
for i in range(0,4):
    min_index, min_value = min(enumerate(MAE_mean_all), key=operator.itemgetter(1));
    regressorAtual = names[min_index]
    if(regressorAtual == 'svm'):
        print "procurando os melhores parametros para o SVM"
        for train_index_skf, test_index_skf in skf.split(X_data_train):
            X_train_skf, X_test_skf = \
                X_data_train[train_index_skf], X_data_train[test_index_skf];

            Y_train_skf, Y_test_skf = \
                values[train_index_skf], values[test_index_skf];

            print "Fazendo o GridSearsh para o SVM regressor...."
            clf = GridSearchCV(svmObj, param_grid=parameters_svm, n_jobs=n_cores, scoring='neg_mean_absolute_error')
            clf.fit(X_train_skf, Y_train_skf)
            print "Finalizado o GridSearsh para o SVM regressor."
            print "MAE do SVM obtido para o conjunto de treino: " + str(-clf.best_score_) + "com os parametros: " + str(
            clf.best_params_)
            svmPreditcTestKfold = clf.predict(X_test_skf);
            MAE_SVM = mean_absolute_error(Y_test_skf, svmPreditcTestKfold)
            print "MAE_SVM obtido para o conjunto de teste: " + str(MAE_SVM)
            if (MAE_SVM <  bestScore_svm):
                bestScore_svm = MAE_SVM
                svm_c_best = clf.best_params_['C']
                svm_gamma_best = clf.best_params_['gamma']
                svm_kernel_best = clf.best_params_['kernel']
            print "melhor SVM parametros ate o momento : " + str(bestScore_svm) + " C: " + str(svm_c_best) + \
                  " gamma: " + str(svm_gamma_best) + " kernel: " + str(svm_kernel_best)

        print "\ncriando o regressor svm com os melhores parametros"
        svmRegression = SVR(C=svm_c_best, kernel=svm_kernel_best, gamma=svm_gamma_best);
        svmRegression.fit(X_data_train, values)
        print "computando a acuracia na base de treino....."
        dados_predicted = svmRegression.predict(X_data_train);
        MAE_SVM = mean_absolute_error(values, dados_predicted)
        print "MAE obtido svm foi de: " + str(MAE_SVM)
        print "aplicado svm regressao nos dados de testes do professor...."
        dadosProfessor_predicted = svmRegression.predict(X_data_test_professor);
        fname = "SVM" + str(i+1)
        np.savetxt(fname=fname, delimiter=',', X=dadosProfessor_predicted, fmt='%f')
        print "-----------fim SVM---------------"
    if(regressorAtual == 'rf'):
        print "procurando os melhores parametros para o RF"
        for train_index_skf, test_index_skf in skf.split(X_data_train):
            X_train_skf, X_test_skf = \
                X_data_train[train_index_skf], X_data_train[test_index_skf];

            Y_train_skf, Y_test_skf = \
                values[train_index_skf], values[test_index_skf];
            print "Fazendo o GridSearsh para o RF regressor...."
            clf2 = GridSearchCV(randomForestObj, param_grid=parameters_rf, n_jobs=n_cores, scoring='neg_mean_absolute_error')
            clf2.fit(X_train_skf, Y_train_skf)
            print "Finalizado o GridSearsh para o RF regressor."
            print "MAE do RF obtido para o conjunto de treino: " + str(-clf2.best_score_) + "com os parametros: " + str(
            clf2.best_params_)
            rfPreditcTestKfold = clf2.predict(X_test_skf);
            MAE_RF = mean_absolute_error(Y_test_skf, rfPreditcTestKfold)
            print "MAE_RF obtido para o conjunto de teste: " + str(MAE_RF)
            if (MAE_RF <  bestScore_rf):
                bestScore_rf = MAE_RF
                rf_n_estimators_best = clf2.best_params_['n_estimators']
                rf_max_features_best = clf2.best_params_['max_features']
                rf_max_depth_best = clf2.best_params_['max_depth']
            print "melhor RF parametros ate o momento: " + str(bestScore_rf) + " n_estimators: " + str(rf_n_estimators_best) \
                  + " max_features: " + str(rf_max_features_best) + " max_depth: " + str(rf_max_depth_best)

        print "\ncriando o regressor rf com os melhores parametros"
        rfRegression = RandomForestRegressor(n_estimators=rf_n_estimators_best,max_features=rf_max_features_best,
                                             max_depth=rf_max_depth_best);
        rfRegression.fit(X_data_train,values)
        print "computando a acuracia na base de treino....."
        dados_predicted = rfRegression.predict(X_data_train);
        MAE_RF = mean_absolute_error(values, dados_predicted)
        print "MAE obtido rf foi de: " + str(MAE_RF)
        print "aplicado o rf regressao nos dados de testes do professor...."
        dadosProfessor_predicted = rfRegression.predict(X_data_test_professor);
        fname = "RF" + str(i+1)
        np.savetxt(fname=fname, delimiter=',', X=dadosProfessor_predicted, fmt='%f')
        print "-----------fim RF---------------"
    if(regressorAtual == 'gbm'):
        print "procurando os melhores parametros para GBM"
        for train_index_skf, test_index_skf in skf.split(X_data_train):
            X_train_skf, X_test_skf = \
                X_data_train[train_index_skf], X_data_train[test_index_skf];

            Y_train_skf, Y_test_skf = \
                values[train_index_skf], values[test_index_skf];
            print "Fazendo o GridSearsh para o Gradient Boosting regressor...."
            clf3 = GridSearchCV(gbmObj, param_grid=parameters_gbm, scoring='neg_mean_absolute_error', n_jobs=n_cores)
            clf3.fit(X_train_skf, Y_train_skf)
            print "Finalizado o GridSearsh para a Gradient Boosting regressor."
            print "MAE do Gradient Boosting obtido para o conjunto de treino: " + str(
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
                  " learning_rate: " + str(gbm_learning_rate_best) + " loss: " + str(
                gbm_loss_best) + " warm_start: " + "True"

        print "\ncriando o regressor gbm com os melhores parametros"
        gbmRegression = GradientBoostingRegressor(n_estimators=gbm_n_estimators_best,max_features=gbm_max_features_best,
                                             max_depth=gbm_max_depth_best,learning_rate=gbm_learning_rate_best);
        gbmRegression.fit(X_data_train,values)
        print "computando a acuracia na base de treino....."
        dados_predicted = gbmRegression.predict(X_data_train)
        MAE_GBM = mean_absolute_error(values, dados_predicted)
        print "MAE obtido gbm foi de: " + str(MAE_GBM)
        print "aplicando o gbm nos dados de testes do professor...."
        dadosProfessor_predicted = gbmRegression.predict(X_data_test_professor);
        fname = "GBM" + str(i+1)
        np.savetxt(fname=fname, delimiter=',', X=dadosProfessor_predicted, fmt='%f')
        print "-----------fim GBM---------------"
    if(regressorAtual == 'mlp'):
        print "procurando os melhores parametros para MLP"
        for train_index_skf, test_index_skf in skf.split(X_data_train):
            X_train_skf, X_test_skf = \
                X_data_train[train_index_skf], X_data_train[test_index_skf];

            Y_train_skf, Y_test_skf = \
                values[train_index_skf], values[test_index_skf];

            print "Fazendo o GridSearsh para a MLP regressor...."
            clf4 = GridSearchCV(mlpObj, param_grid=parameters_mlp, n_jobs=n_cores, scoring='neg_mean_absolute_error')
            clf4.fit(X_train_skf, Y_train_skf)
            print "Finalizado o GridSearsh para a MLP regressor."
            print "MAE da MLP obtido para o conjunto de treino: " + str(
                -clf4.best_score_) + " com os parametros: " + str(clf4.best_params_)
            mlpPreditcTestKfold = clf4.predict(X_test_skf);
            MAE_MLP = mean_absolute_error(Y_test_skf, mlpPreditcTestKfold)
            print "MAE_MLP obtido para o conjunto de teste: " + str(MAE_MLP)
            if (MAE_MLP < bestScore_mlp):
                bestScore_mlp = MAE_MLP
                mlp_hidden_layer_sizes_best = clf4.best_params_['hidden_layer_sizes']
                mlp_solver_best = clf4.best_params_['solver']
                mlp_activation_best = clf4.best_params_['activation']
                mlp_learning_rate_best = clf4.best_params_['learning_rate']
                mlp_learning_rate_init_best = clf4.best_params_['learning_rate_init'];
            print "melhor MLP parametros ate o momento: " + str(bestScore_mlp) + " hidden_layer_sizes: " + str(mlp_hidden_layer_sizes_best) \
                  + " solver: " + str(mlp_solver_best) + " activation: " + str(mlp_activation_best) + \
                  " learning_rate: " +  str(mlp_learning_rate_best) + " learning_rate_init: " + str(mlp_learning_rate_init_best)

        print "\ncriando o regressor mlp com os melhores parametros"
        mlpRegression = MLPRegressor(hidden_layer_sizes=mlp_hidden_layer_sizes_best,solver=mlp_solver_best,
                                                  activation=mlp_activation_best,learning_rate=mlp_learning_rate_best,
                                                  learning_rate_init=mlp_learning_rate_init_best);
        mlpRegression.fit(X_data_train,values)
        print "computando a acuracia na base de treino....."
        dados_predicted = mlpRegression.predict(X_data_train);
        MAE_MLP = mean_absolute_error(values, dados_predicted)
        print "MAE obtido mlp foi de: " + str(MAE_MLP)
        print "aplicando a mlp nos dados de testes do professor...."
        dadosProfessor_predicted = mlpRegression.predict(X_data_test_professor);
        fname = "MLP" + str(i+1)
        np.savetxt(fname=fname, delimiter=',', X=dadosProfessor_predicted, fmt='%f')
        print "-----------fim MLP---------------"
    del names[min_index]
    del MAE_mean_all[min_index]
print "---------------------fim de execucao---------------------"

# svmObj = SVR(C=svm_c_best,gamma=svm_gamma_best,kernel=svm_kernel_best);
# svmObj.fit(X_data_train,values);
# values_predicted = svmObj.predict(X_data_train);
# score_svm = mean_absolute_error(values,values_predicted)
#
# randomForestObj = RandomForestClassifier(n_estimators=n_estimators_best,
#                                          max_features=max_features_best,
#                                          max_depth=max_depth_best)
# randomForestObj.fit(X_data_train,values);
# values_predicted = randomForestObj.predict(X_data_train)
# score_rf = mean_absolute_error(values,values_predicted)
# Y_predict = []
# Y_predict_notbest = []
# fname_best = ''
# fname_notbest = ''
# print "svm_score_final: " + str(score_svm) + " rf_score_final: " + str(score_rf)
# if(score_svm < score_rf):
#     Y_predict = svmObj.predict(X_data_train)
#     fname_best = 'testPredictSVM_best.txt'
#     Y_predict_notbest = randomForestObj.predict(X_data_train)
#     fname_notbest = 'testPredictRF_notbest.txt'
# else:
#     Y_predict = randomForestObj.predict(X_data_train)
#     fname_best = 'testPredictRF_best.txt'
#     Y_predict_notbest = svmObj.predict(X_data_train)
#     fname_notbest = 'testPredictSVM_notbest.txt'
#
# np.savetxt(fname=fname_best,delimiter=',',X=Y_predict,fmt='%f')
# np.savetxt(fname=fname_notbest,delimiter=',',X=Y_predict_notbest,fmt='%f')
