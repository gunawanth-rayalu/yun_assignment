import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
#we are using clustering method to group points
x = {
    'x': [1,1.5,2,2.5,3,3.5,4,4.5],
    'y': [1,1,1.5,2,2.5,2,2,2]
}
df = pd.DataFrame(x)
cluster_maker = DBSCAN(eps = 0.5, min_samples=1)
#groups all points which are in range of 0.5 to each other and min_samples = 1 to consider all points
df['group'] = cluster_maker.fit_predict(df[['x', 'y']])
count = df['group'].value_counts()
grouped_df = df[df['group'].isin(count[count > 1].index)]

