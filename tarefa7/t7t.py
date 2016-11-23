import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import operator

def ReadData(file):
    serie = np.genfromtxt(file, delimiter=',', dtype='str')
    X = serie[1:,1]
    X = X.astype(float)
    return X;
#------------------------------------------------------#

def computarN(X, threshold):
    bubbles_c = np.array([])
    bubbles_n = np.array([])
    firstBubble=0;
    firstIteration=True;

    for index in range(0, len(X)):

        if index == 0:
            bubbles_c = np.concatenate((bubbles_c,X[index:index+1]),axis=0)
            bubbles_n = np.concatenate((bubbles_n,[1]),axis=0)
            firstBubble = 0;

        else:
            closerBubble = abs(bubbles_c - X[index]);
            min_index, min_value = min(enumerate(closerBubble), key=operator.itemgetter(1));
            if (min_value <= threshold):
                if min_index == firstBubble  and (not(firstIteration)):
                    break
                else:
                    bubbles_c[min_index] = ((bubbles_c[min_index]*bubbles_n[min_index]) +  X[index])/(bubbles_n[min_index]+1)
                    bubbles_n[min_index] = bubbles_n[min_index] + 1

            else:
                bubbles_c = np.concatenate((bubbles_c, X[index:index + 1]), axis=0)
                bubbles_n = np.concatenate((bubbles_n, [1]), axis=0)
                firstIteration = False
    bubbles_c2 = np.array([])
    bubbles_n2 = np.array([])
    firstBubble=0;
    firstIteration=True;
    for iter in range(index, len(X)):

        if iter == index:
            bubbles_c2 = np.concatenate((bubbles_c2,X[iter:iter+1]),axis=0)
            bubbles_n2 = np.concatenate((bubbles_n2,[1]),axis=0)
            firstBubble = 0;
        else:
            closerBubble = abs(bubbles_c2 - X[iter]);
            min_index, min_value = min(enumerate(closerBubble), key=operator.itemgetter(1));
            if (min_value <= threshold):
                if min_index == firstBubble  and (not(firstIteration)):
                    break
                else:
                    bubbles_c2[min_index] = ((bubbles_c2[min_index]*bubbles_n2[min_index]) +  X[index])/(bubbles_n2[min_index]+1)
                    bubbles_n2[min_index] = bubbles_n2[min_index] + 1

            else:
                bubbles_c2 = np.concatenate((bubbles_c2, X[iter:iter + 1]), axis=0)
                bubbles_n2 = np.concatenate((bubbles_n2, [1]), axis=0)
                firstIteration = False

    return sum(bubbles_n2)

def acharAnomalia(X):
    dataStd = np.std(X);
    print "deviation: " + str(dataStd)
    N = computarN(X,dataStd)
    N = int(N)
    print "N: " + str(N)

    p = []
    #computa a media e variancia para o primeiro intervalo
    p.append([np.mean(X[0:N]), np.std(X[0:N])])    
    i = N
    # computa a media e variancia para o cada intervalo e concatena isso em p
    while(i <= len(X) - N):
        p.append([np.mean(X[i:i + N]), np.std(X[i:i + N])])
        i += N


        
    neighbours = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(p)
    dist, ind = neighbours.kneighbors(p)
    
    mediaDist = np.mean(dist[:,1:])
    desvioDist = np.std(dist[:,1:])
    
    for k in range(len(dist)):
        inicio = k*N
        if(inicio + N > len(X)):
            fim = len(X)
        else:
            fim = inicio + N - 1
        ran = list(range(inicio, fim))
        if(dist[k][1] > (mediaDist - desvioDist) and dist[k][1] < (mediaDist + desvioDist)):        
            plt.plot(ran, X[inicio:fim],color=(0,0,1))
        else:
            plt.plot(ran, X[inicio:fim],color=(1,0,0))
    plt.show()

#encontra as anomalias
t1 = 15.0
s1 = ReadData('serie1.csv')
acharAnomalia(s1)

s2 = ReadData('serie2.csv')
acharAnomalia(s2)

s3 = ReadData('serie3.csv')
acharAnomalia(s3)

s4 = ReadData('serie4.csv')
acharAnomalia(s4)
t2 = 1.5
s5 = ReadData('serie5.csv')
acharAnomalia(s5)
#------------------------------------------------------#
    
