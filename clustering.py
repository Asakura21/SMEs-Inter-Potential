import pandas as pd   # import pandas
import numpy as np
import seaborn as sb
import matplotlib
import matplotlib.pyplot as plt
from minisom import MiniSom
from sklearn.preprocessing import MinMaxScaler

# Initialization and training
n_neurons = 10
m_neurons = 10
som = MiniSom(n_neurons, m_neurons, df_scale.shape[1], sigma=1.5, learning_rate = 0.5, neighborhood_function='gaussian', random_seed=20)
som.random_weights_init(df_scale)
som.train_random(data = df_scale, num_iteration = 10000, verbose=True)

from pylab import bone, pcolor, colorbar, plot, show
%matplotlib inline 
plt.figure(figsize=(5, 5), dpi=100)
bone()
pcolor(som.distance_map().T)
colorbar()
plt.show()


#  finding the cluster
mappings = som.win_map(df_scale)  # returns a dictionary mapping the winning nodes to the customers
print(np.shape(mappings[(2, 2)]))
unusual_observations2 = mappings[(2, 2)]

# Finding the cluster
#mappings = som.win_map(X)
unusual_observations2 = sc.inverse_transform(unusual_observations2)
out = pd.DataFrame(unusual_observations2)
out.columns = df.columns
out

#  finding the cluster
print(np.shape(mappings[(2, 2)]))
len(mappings)

# creating the dependent variable
is_bmu = []
for i in range(len(df_scale)):
    # find its Best Matching Unit
    is_bmu.append(som.winner(df_scale[i]))

is_bmu[0:10]

mappings.keys()

cluster_c = []
cluster_n = []
for x in range(n_neurons):
    for y in range(m_neurons):
        cluster = (x,y)
        if cluster in mappings.keys():
            cluster_c.append(len(mappings[cluster]))
        else:
            cluster_c.append(0)
        cluster_number = x*m_neurons+y+1
        cluster_n.append(f"Cluster {cluster_number}")

plt.figure(figsize=(25,5))
plt.title("Food & Ratio: Cluster Distribution for SOM")
plt.bar(cluster_n, cluster_c)

#plt.savefig('Cluster_FoodRatio_SOM.png')
plt.show()


namesofMySeries = df.columns.to_list()
cluster_map = []
for idx in range(len(df_scale)):
    winner_node = som.winner(df_scale[idx])
    cluster_map.append(([idx],f"Cluster {winner_node[0]*m_neurons+winner_node[1]+1}"))
    
pd.DataFrame(cluster_map,columns=["Series","Cluster"]).sort_values(by="Cluster").set_index("Series")

clus = pd.DataFrame(cluster_map,columns=["Series","Cluster"])
df_clus = df
df_clus['Cluster'] = clus['Cluster']
df_mean = df_clus.groupby('Cluster').mean()
df_mean_T = df_clus.groupby('Cluster').mean().T
df_mean

df_count = df_clus.groupby('Cluster').count().T
df_count
