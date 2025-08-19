from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
data = pd.read_csv("Task 3 and 4_Loan_Data.csv")
target1data = data[data['fico_score']< 600]
target2data = data[data['fico_score'] >= 600 and data['fico_score'] <= 850]

kmeans_under_600 = KMeans(n_clusters=5, random_state=0).fit(target1data.values)
kmeans_between_600_and_850 = KMeans(n_clusters=5, random_state=0).fit(target2data.values)


def label_buckets(data):
    if data < 600:
        cluster = kmeans_under_600.predict([[data]])[0]
        ans = 9 - cluster  
    else:
        ans = kmeans_between_600_and_850.predict([[data]])[0]
    return ans
















