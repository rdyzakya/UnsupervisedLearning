import numpy as np
import time

class DistanceMethodNotValidError(Exception):
	pass

class NotSameLength(Exception):
	pass

class KMedoidClustering:
	def __init__(self):
		self.medoids = None
		self.x_columns = None
		self.how = None
		self.df = None

	def euclidean_distance(self,this_row,other):
		res = 0
		for cols in self.x_columns:
			delta = this_row[cols] - other[cols]
			delta_sqr = delta**2
			res += delta_sqr

		return np.sqrt(res)

	def manhattan_distance(self,this_row,other):
		res = 0
		for cols in self.x_columns:
			delta = this_row[cols] - other[cols]
			delta_abs = np.abs(delta)
			res += delta_abs

		return res

	def calculate_nearest(self,row,how='euclidean'):
		dist = [0 for i in range(len(self.medoids))]
		dist = np.array(dist)
		for i in range(len(self.medoids)):
			if how == 'euclidean':
				dist[i] = self.euclidean_distance(row,self.medoids.loc[i])
			elif how == 'manhattan':
				dist[i] = self.manhattan_distance(row,self.medoids.loc[i])
			else:
				raise DistanceMethodNotValidError()
		min_idx = np.where(dist == dist.min())[0][0]
		return min_idx

	def cost(self,o,df,how='euclidean'):
		if how == 'euclidean':
			df['Distance'] = self.euclidean_distance(df[self.x_columns],o)
		elif how == 'manhattan':
			df['Distance'] = self.manhattan_distance(df[self.x_columns],o)
		else:
			raise DistanceMethodNotValidError()
		res = df['Distance'].sum()
		del df['Distance']
		return res

	def train(self,df_,x_columns,k,how='euclidean'):
		start = time.time()
		df = df_.copy()
		self.x_columns = [df.columns[i] for i in x_columns]
		self.medoids = df.sample(k).copy()
		self.medoids = self.medoids.reset_index()
		self.medoids = self.medoids[self.x_columns]
		self.how = how
		df['Label'] = np.nan
		df['New Label'] = np.nan
		while False in (df['Label'] == df['New Label']).unique():
			df['Label'] = df.apply(lambda row: self.calculate_nearest(row[self.x_columns],self.how),axis=1)
			for i in range(len(self.medoids)):
				cluster = df[df['Label'] == i].copy()
				cluster['Cost'] = cluster.apply(lambda row: self.cost(row[self.x_columns],cluster,self.how),axis=1)
				idxmin = cluster[['Cost']].idxmin().values[0]
				del cluster['Cost']
				self.medoids.loc[i] = cluster.loc[idxmin].copy()
			df['New Label'] = df.apply(lambda row: self.calculate_nearest(row[self.x_columns],self.how),axis=1)

		df['Label'] = df['New Label']
		del df['New Label']
		self.df = df
		print(time.time() - start)

	def predict(self,data):
		if len(self.x_columns) != len(data):
			raise NotSameLength()

		temp = data
		data = {}
		for i in range(len(self.x_columns)):
			data[self.x_columns[i]] = temp[i]

		return self.calculate_nearest(data,self.how)