import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score, calinski_harabaz_score, adjusted_rand_score, adjusted_mutual_info_score, homogeneity_completeness_v_measure
import matplotlib.pyplot as plt

data = []
label = []
print "\n---------------------------------------------------"
#le os dados de entrada
data = np.loadtxt('cluster-data.csv', delimiter = ',', skiprows=1);
label = np.loadtxt('cluster-data-class.csv', skiprows=1);

X = np.array(data);
Y = np.array(label);
#da um shift nos labes: [1-7] para [0-6]
Y -= 1;

range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10]

k_vec = []
silhouette_vec = [];
calinski_harabaz_vec = [];

#avalia a metrica interna para escolher o k
for n_clusters in range_n_clusters:
  clusterer = KMeans(n_clusters=n_clusters,  n_init=5)
  cluster_labels = clusterer.fit_predict(X)
  #silhouette_avg = silhouette_score(X, cluster_labels)
  # silhouette_vec.append(silhouette_avg);
  calinski_harabaz_avg = calinski_harabaz_score(X,cluster_labels)
  calinski_harabaz_vec.append(calinski_harabaz_avg);
  k_vec.append(n_clusters);

  #print("For n_clusters =", n_clusters,"The average silhouette_score is :", silhouette_avg)
  print("For n_clusters =", n_clusters, \
        "The average calinski_harabaz_score is :", calinski_harabaz_avg)


#frescuras para plotar o grafico
myVec = calinski_harabaz_vec;
ind = np.arange(9)
ind = ind
width = 0.7
high = max(myVec)
fig, ax = plt.subplots()
rects1 = ax.bar(ind, myVec, width, color='b')
ax.set_ylabel('Scores')
ax.set_title('metrica interna (Calinski-Harabaz index)')
width2 = width/2
ax.set_xticks(ind + width2)
ax.set_xticklabels(('2', '3', '4', '5', '6', '7', '8', '9', '10'))

def autolabel(rects):
    # attach some text labels
    for rect in rects:
      height = rect.get_height()
      ax.text(rect.get_x() + rect.get_width()/2, 1.05*height,
        '%d' % int(height),
        ha='center', va='bottom')

autolabel(rects1)
axes = plt.gca()
axes.set_ylim([0,high*1.1])
plt.xlabel('k')
plt.plot();

print "\n"
#avalia a metrica externa avaliar o kmeans
adjRand_index_vec = []
for n_clusters in range_n_clusters:
  clusterer = KMeans(n_clusters=n_clusters,  n_init=5)
  cluster_labels = clusterer.fit_predict(X)
  adjRand_index = adjusted_rand_score(Y,cluster_labels)
  #adjRand_index = adjusted_mutual_info_score(Y, cluster_labels)
  #print adjRand_index
  adjRand_index_vec.append(adjRand_index)
  print("For n_clusters =", n_clusters, "The Adjusted Rand index is :", adjRand_index)

#frescuras para plotar o segundo grafico
myVec = adjRand_index_vec;
ind = np.arange(9)
ind = ind
width = 0.7
high = max(myVec)
fig2, ax2 = plt.subplots()
rects2 = ax2.bar(ind, myVec, width, color='b')

# add some text for labels, title and axes ticks
ax2.set_ylabel('Scores')
ax2.set_title('metrica externa (Adjusted Rand index)')
width2 = width/2
ax2.set_xticks(ind + width2)
ax2.set_xticklabels(('2', '3', '4', '5', '6', '7', '8', '9', '10'))

def autolabelF(rects):
    # attach some text labels
    for rect in rects:
      height = rect.get_height()
      ax2.text(rect.get_x() + rect.get_width()/2, 1.05*height,
        '%0.3f' % float(height),
        ha='center', va='bottom')

autolabelF(rects2)
axes2 = plt.gca()
axes2.set_ylim([0,high*1.1])
plt.xlabel('k')
plt.plot()
plt.draw()
plt.show()