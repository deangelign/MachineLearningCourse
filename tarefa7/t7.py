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
import matplotlib.pyplot as plt
import operator

values = np.loadtxt('serie1.csv', delimiter = ',', usecols=(1,),skiprows=1);
indices = np.array(range(0,len(values)))

data = np.column_stack((indices,values))

# plt.plot(data[:, 0],data[:, 1], color="blue")
# plt.plot(data[:, 0],2*data[:, 1], color="red")
# plt.show()

iteration = 0
bubbles_c = np.array([])
bubbles_n = np.array([])
bubbles_r = np.array([])
bubbles_r_default = 10;
numberMinOutliers = 5;

outliers = np.array([])

res = abs(data[:,1] - np.array([100]))
min_index, min_value = min(enumerate(res), key=operator.itemgetter(1));

#ex = np.array([True ,False, False, False, True, True, False])
#index = np.nonzero(ex)
#print index
#print len(index[0])
#print ex
#print ex[index]
#print np.sum(ex[index])
#ex = np.delete(ex,index)
#print ex

#print data[:,1]
#print res

iter = 0
iterMax = numberMinOutliers
while True:
    iteration = 0;
    for row in data:
        #print "miau"
        if iteration == 0:
            #print len(bubbles_c)
            #print len(values[iteration:iteration+1])
            bubbles_c = np.concatenate((bubbles_c,values[iteration:iteration+1]),axis=0)
            bubbles_n = np.concatenate((bubbles_n,[1]),axis=0)
            bubbles_r = np.concatenate((bubbles_r,[bubbles_r_default]),axis=0)
        else:
            closerBubble = abs(bubbles_c - values[iteration])
            min_index, min_value = min(enumerate(closerBubble), key=operator.itemgetter(1));
            if(min_value <= bubbles_r_default):
                bubbles_c[min_index] = ((bubbles_c[min_index]*bubbles_n[min_index]) +  values[iteration])/(bubbles_n[min_index]+1)
                bubbles_n[min_index] = bubbles_n[min_index] + 1
            else:
                outliers = np.concatenate((outliers,values[iteration:iteration+1]), axis=0)
                closerOutliers = abs(outliers - values[iteration]) < bubbles_r_default
                index_closerOutliers = np.nonzero(closerOutliers)
                n_close_outlier = len(index_closerOutliers[0])
                if n_close_outlier > numberMinOutliers:
                    new_bubble_location = [np.sum(outliers[index_closerOutliers])/n_close_outlier]
                    bubbles_c = np.concatenate((bubbles_c, new_bubble_location),axis=0)
                    bubbles_n = np.concatenate((bubbles_n,[n_close_outlier]),axis=0)
                    bubbles_r = np.concatenate((bubbles_r, [bubbles_r_default]), axis=0)
                    outliers = np.delete(outliers, index_closerOutliers, None)
        outliers_i = 0
        while(outliers_i < len(outliers)):
            closerBubble = abs(bubbles_c - outliers[outliers_i])
            min_index, min_value = min(enumerate(closerBubble), key=operator.itemgetter(1));
            if (min_value <= bubbles_r_default):
                bubbles_c[min_index] = ((bubbles_c[min_index] * bubbles_n[min_index]) + values[iteration]) / (
                    bubbles_n[min_index] + 1)
                bubbles_n[min_index] = bubbles_n[min_index] + 1
                outliers = np.delete(outliers, outliers_i, None)
                outliers_i = 0
            else:
                outliers_i = outliers_i + 1


        iteration += 1
        # print "iteracao " + str(iteration-1)
        # print "point: " + str(values[iteration-1])
        # print "centroides: " + str(bubbles_c.tolist())
        # print "n: " + str(bubbles_n.tolist())
        # print "outliers: " + str(outliers)
        # print "---------------------------------------------------------------"
        #print bubbles_c.tolist()

    iter = iter + 1
    print "--------" + str(iter) + "----------------------------------\n"
    print len(outliers)
    if(len(outliers)  == 0 or iter > iterMax):
        break;



#nice print
# order_indices = np.argsort(bubbles_n)
# bubbles_n_sorted = bubbles_n[order_indices]
# bubbles_c_sorted = bubbles_c[order_indices]
# bubbles_n_sorted = bubbles_n_sorted[::-1]
# bubbles_c_sorted = bubbles_c_sorted[::-1]
# colorPower = bubbles_n_sorted/bubbles_n_sorted[0]
# color_poins = []
#
# for index in range(0,len(values)):
#     closerBubble = abs(bubbles_c_sorted - values[index])
#     min_index, min_value = min(enumerate(closerBubble), key=operator.itemgetter(1));
#     plt.plot(indices[index],values[index],'o',color=(colorPower[min_index],0,0))
#
# print bubbles_c_sorted
# print bubbles_n_sorted
# plt.show()
#
order_indices = np.argsort(bubbles_c)
bubbles_n_sorted = bubbles_n[order_indices]
bubbles_c_sorted = bubbles_c[order_indices]
bubbles_r_sorted = bubbles_r[order_indices]

#merge bubbles
bubble_i = 0
while(bubble_i < len(bubbles_c_sorted)-1):

    distanceClosest = bubbles_c_sorted[bubble_i+1] - bubbles_c_sorted[bubble_i]
    if distanceClosest < bubbles_r_sorted[bubble_i]:
        newNPoints = bubbles_n_sorted[bubble_i] + bubbles_n_sorted[bubble_i+1]
        newCentroidLocation = (
                                  (bubbles_c_sorted[bubble_i]*bubbles_n_sorted[bubble_i]) + \
                                  (bubbles_c_sorted[bubble_i+1]*bubbles_n_sorted[bubble_i+1])
                              ) \
                              /(newNPoints)
        # minimumVariationB1 = abs(bubbles_c_sorted[bubble_i] - bubbles_r_sorted[bubble_i])
        # minimumVariationB2 = abs(bubbles_c_sorted[bubble_i+1] - bubbles_r_sorted[bubble_i+1])
        # MaximumVariationB1 = abs(bubbles_c_sorted[bubble_i] + bubbles_r_sorted[bubble_i])
        # MaximumVariationB2 = abs(bubbles_c_sorted[bubble_i + 1] + bubbles_r_sorted[bubble_i + 1])
        delta = max(newCentroidLocation-bubbles_c_sorted[bubble_i],bubbles_c_sorted[bubble_i+1]-newCentroidLocation)

        newRadius = max(bubbles_r_sorted[bubble_i],bubbles_r_sorted[bubble_i+1])+delta

        bubbles_c_sorted = np.delete(bubbles_c_sorted, [bubble_i,bubble_i+1], None)
        bubbles_n_sorted = np.delete(bubbles_n_sorted, [bubble_i, bubble_i + 1], None)
        bubbles_r_sorted = np.delete(bubbles_r_sorted, [bubble_i, bubble_i + 1], None)

        bubbles_c_sorted = np.insert(arr=bubbles_c_sorted,obj=0,values=newCentroidLocation)
        bubbles_n_sorted = np.insert(arr=bubbles_n_sorted, obj=0, values=newNPoints)
        bubbles_r_sorted = np.insert(arr=bubbles_r_sorted, obj=0, values=newRadius)
        bubble_i = 0
    else:
        bubble_i = bubble_i + 1

#nice print
# order_indices = np.argsort(bubbles_n_sorted)
# bubbles_n_sorted2 = bubbles_n_sorted[order_indices]
# bubbles_c_sorted2 = bubbles_c_sorted[order_indices]
# bubbles_n_sorted2 = bubbles_n_sorted2[::-1]
# bubbles_c_sorted2 = bubbles_c_sorted2[::-1]
# colorPower = bubbles_n_sorted2/bubbles_n_sorted2[0]
#
#
# for index in range(0,len(values)):
#     closerBubble = abs(bubbles_c_sorted2 - values[index])
#     min_index, min_value = min(enumerate(closerBubble), key=operator.itemgetter(1));
#     plt.plot(indices[index],values[index],'o',color=(colorPower[min_index],0,0))
#
# plt.show()
closerBubble = abs(bubbles_c_sorted - values[0])
min_index, min_value = min(enumerate(closerBubble), key=operator.itemgetter(1));
acc = 1
last_bubbles = np.array([[min_index],[1],[acc]])
acc += 1
frequencyChange = np.zeros(shape=(len(bubbles_c_sorted),len(bubbles_c_sorted)))



bubble_i = 0
for index in range(1,len(values)):
    closerBubble = abs(bubbles_c_sorted - values[index])
    min_index, min_value = min(enumerate(closerBubble), key=operator.itemgetter(1));
    if min_index == last_bubbles[0,bubble_i]:
        last_bubbles[1,bubble_i] = last_bubbles[1,bubble_i] + 1
        last_bubbles[2, bubble_i] = acc
        acc += 1
    else:
        frequencyChange[last_bubbles[0, bubble_i], min_index] += 1
        aux = np.array([[min_index], [1],[acc]])
        last_bubbles = np.concatenate((last_bubbles, aux),axis=1)
        bubble_i = bubble_i+1
        acc += 1


last_bubbles = last_bubbles.T
probChange = frequencyChange
t = 0.1
for i in range(0,len(bubbles_c_sorted)):
    probChange[i,:] = frequencyChange[i,:]/sum(frequencyChange[i,:])


closerBubble = abs(bubbles_c_sorted - values[0])
min_index, min_value = min(enumerate(closerBubble), key=operator.itemgetter(1));
flag_anomaly = False
last_index = min_index
color = 0
for index in range(1,len(values)):

    closerBubble = abs(bubbles_c_sorted - values[index])
    min_index, min_value = min(enumerate(closerBubble), key=operator.itemgetter(1));
    if flag_anomaly:
        if last_index == min_index:
            flag_anomaly = False
            plt.plot(indices[index], values[index], 'o', color=(1, 0, 0))
        else:
            plt.plot(indices[index], values[index], 'o', color=(0, 0, 0))
    else:
        if min_index == last_index:
            if flag_anomaly:
                plt.plot(indices[index], values[index], 'o', color=(0, 0, 0))
            else:
                plt.plot(indices[index], values[index], 'o', color=(1, 0, 0))

        else:
            if probChange[last_index,min_index] > t:
                last_index = min_index
            else:
                flag_anomaly = True

            plt.plot(indices[index], values[index], 'o', color=(1, 0, 0))


plt.show()