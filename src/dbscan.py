import numpy as np

class Flag:
	def __init__(self):
		self.flag = None

	def set_false(self):
		self.flag = False

	def set_true(self):
		self.flag = True

class DBScanClustering:
	def __init__(self):
		self.df = None
		self.x_columns = None

	def euclidean_distance(self,this_row,other):
		res = 0
		for cols in self.x_columns:
			delta = this_row[cols] - other[cols]
			delta_sqr = delta**2
			res += delta_sqr
		return np.sqrt(res)

	def train(self,df,x_columns,eps,min_point):
		self.x_columns = [df.columns[i] for i in x_columns]
		self.df = df.copy()
		self.df['visited'] = 0
		self.df['Label'] = np.nan
		cluster_count = 0
		stack = []
		f = Flag()
		while 0 in self.df['visited'].values:
			core_idx = self.df[self.df['visited'] == 0].index[0]
			stack.append(core_idx)
			f.set_false()
			while len(stack) != 0:
				self.clustering(self.df,cluster_count,eps,min_point,stack,f)
			if f.flag:
				cluster_count += 1
		self.df.loc[self.df['Label'].isna(), 'Label'] = -1
		self.df['Label'] = self.df.Label.astype(int)
		del self.df['visited']


	def clustering(self,df,label,eps,min_point,stack,f):
		idx = stack.pop()
		df.loc[idx,'visited'] = 1
		df['distance'] = self.euclidean_distance(df,df.loc[idx])
		condition = df['Label'].isna() | (df['Label'] == label)
		surround = df[(df['distance'] < eps) & condition]
		if len(surround) >= min_point:
			df.loc[(df['distance'] < eps) & df['Label'].isna(), 'Label'] = label
			del df['distance']
			for i in surround.index:
				if df.loc[i,'visited'] == 0:
					stack.append(i)
			if not f.flag:
				f.set_true()
