# Unsupervised Learning
Repositori ini berisikan model clustering sederhana yakni K-Means Clustering, K-Medoids Clustering, dan DBSCAN.

## How To Use
Berikut adalah tata cara pemakaian model unsupervised learning

1. Pastikan Python telah tersedia (dibuat dan dites menggunakan Python 3.8.5)
2. Pastikan library pandas dan numpy tersedia
3. Buatlah sebuah file notebook (ipynb), bisa juga dipakai di file .py sesuai kebutuhan
4. Lakukan import source code model

```python
from src import kmeans,dbscan,kmedoids
```

5. Instansiasi dataframe menggunakan pandas

```python
import pandas as pd
df = pd.read_csv('namafile.csv')
```

6. Instansiasi Model

```python
clf1 = kmeans.KMeansClustering()
clf2 = kmedoids.KMedoidClustering()
clf3 = dbscan.DBScanClustering()
```

### Penggunaan K-Means dan K-Medoid
7.a. Lakukan clustering pada model menggunakan dataset

```python
#pilih cara kalkulasi costnya, bisa menggunakan euclidean distance
#bisa juga dengan manhattan distance
clf1.fit(df,x_columns=[0,1],k=3,how='euclidean')
clf2.fit(df,x_columns=[0,1],k=3,how='manhattan')
```

7.b. Peroleh hasilnya

```python
print(clf1.df)
print(clf2.df)
```

7.c. Lakukan prediksi

```python
clf1.predict([0.5,0.3])
```

### Penggunaan DBSCAN
8.a. Lakukan clustering pada model dengan dataset

```python
clf3.fit(df,[0,1],eps=0.01,min_point=5)
```

8.b. Peroleh hasilnya

```python
print(clf3.df)
```
