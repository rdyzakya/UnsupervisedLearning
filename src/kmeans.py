import pandas as pd
import numpy as np

class DistanceMethodNotValidError(Exception):
	pass

class NotSameLength(Exception):
	pass

class KMeansClustering:
	def __init__(self,df):
		self.df = df.copy()
		self.centroids = None
		self.x_columns = None
		self.how = None

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
		dist = [0 for i in range(len(self.centroids))]
		dist = np.array(dist)
		for i in range(len(self.centroids)):
			if how == 'euclidean':
				dist[i] = self.euclidean_distance(row,self.centroids.loc[i])
			elif how == 'manhattan':
				dist[i] = self.manhattan_distance(row,self.centroids.loc[i])
			else:
				raise DistanceMethodNotValidError()
		min_idx = np.where(dist == dist.min())[0][0]
		return min_idx


	def train(self,x_columns,k,how='euclidean'):
		self.x_columns = [self.df.columns[i] for i in x_columns]
		self.centroids = self.df.sample(k).copy()
		self.centroids = self.centroids.reset_index()
		self.centroids = self.centroids[self.x_columns]
		self.how = how
		self.df['Label'] = np.nan
		self.df['New Label'] = np.nan
		while False in (self.df['Label'] == self.df['New Label']).unique():
			self.df['Label'] = self.df.apply(lambda row: self.calculate_nearest(row[self.x_columns],how),axis=1)
			for i in range(len(self.centroids)):
				df_i = self.df[self.df['Label'] == i]
				means = df_i.mean()
				for col in self.x_columns:
					self.centroids.loc[i,col] = means[col]
			self.df['New Label'] = self.df.apply(lambda row: self.calculate_nearest(row[self.x_columns],how),axis=1)

		self.df['Label'] = self.df['New Label']
		del self.df['New Label']

	def predict(self,data):
		if len(self.x_columns) != len(data):
			raise NotSameLength()

		temp = data
		data = {}
		for i in range(len(self.x_columns)):
			data[self.x_columns[i]] = temp[i]

		return self.calculate_nearest(data,self.how)


