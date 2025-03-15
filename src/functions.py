# You will notice that my variable names inside function tend to be
# different from the ones I use otherwise style wise. I do this because
# the locality of a function allows for reuse of variable names and I
# really like small simple variable names (you can probably tell)

from typing import Sequence
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections.abc import Callable
from time import time

def timeit(func: Callable) -> Callable:
	"""Prints the time a function took to execute"""
	# This is my favorite functions and if you look at my other repositories on github it becomes quite obvious
	def wrapper(*args, **kwargs):
		start= time()
		result = func(*args, **kwargs)
		end = time()
		elapsed = end - start
		print(f"Function '{func.__name__}' executed in {elapsed:.4f} seconds")
		return result
	return wrapper

@timeit
def dataSplit(dataframe: pd.DataFrame) -> tuple:
	"""Splits the dataframe to data (x's and y's together) and metadata (id, sex...)"""
	# I use hard coded metrics because we have been informed about the format
	# If the format was permiable I'd make this take input and somehow determine it
	# in another function I think
	data = dataframe.iloc[:, 4:]
	metadata = dataframe.iloc[:, 0:4]
	return data, metadata

@timeit
def isolator(data: pd.DataFrame) -> tuple:
	"""Isolates the x's and y's of the dataframe into different variables"""
	x = data.iloc[:, 1:] # everything other than the BMI
	y = data['BMI']
	return x, y

@timeit
def regressions(methods: Sequence[Callable],
				train: pd.DataFrame,
				pred: pd.DataFrame,
				val: pd.DataFrame) -> np.ndarray:
	"""Calls the method on the data given and return outputs"""
	# for no re computations
	scale = len(methods)
	# Variable to keep outputs
	predictions = np.array(
		[
			# Wait what is this isn't this supposed
			# to be a simple structure for keeping
			# the data?
			# Well yeah I want to pre allocate all the memory
			# so I use this format, I also want to use numpy
			# arrays for two reasons 1. they are fast 2. the 
			# output of the regression function is a numpy array
			# So what is this segment benerath?
			# it's me allocating # of samples in the validation set
			# (or in other words number of rows in the val csv)
			# positions in an array, shape gives (rows, cols)
			# so shape[0] = rows
			np.zeros(
				val.shape[0]
			)
		] * scale
	)
	for index, model in enumerate(methods):
		object = model()
		object.fit(train, pred)
		predictions[index] = object.predict(val)
	return predictions

@timeit
def applyMetrics(metrics: Sequence[Callable],
				 options: Sequence[Callable],
				 truth: np.ndarray,
				 data: np.ndarray) -> np.ndarray:
	"""Applies all the given metrics to the data assuming as ground truth the truth given"""
	# Note that the function takes input as numpy array and not a dataframe which is why we convert the input in the notebook in the ()'s of the function
	# The options are the regressors the metrics are the cost functions
	# To not re compute this
	holder = len(options)
	# Variable to keep the metrics
	output = np.array(
		[
			[
				# See previous function for reasoning behind complexity here
				np.array([0])
			] * holder
		] * len(metrics)
	)
	# So we will end up with the following
	# [[option1, option2, option3, ...], # metric1
	# [option1, option2, option3, ...], # metric2
	# [...], # ...3
	# [...], # ...4
	# ...]
	for i, metric in enumerate(metrics):
		for index in range(holder):
			output[i, index] = metric(truth, data[index])
	return output

@timeit
def visualisePredictions(input: np.ndarray,
						 methods: Sequence[Callable],
						 verbose: bool = False) -> None:
	"""Takes the output of the regressions function and displayes it with matplotlib"""
	# I lean heavily on the documentation for this
	# https://matplotlib.org/stable/gallery/statistics/boxplot.html#sphx-glr-gallery-statistics-boxplot-py
	# Wonderful documentation honestly
	holder = len(methods)
	fig, ax = plt.subplots(nrows=1, ncols=holder, figsize=(3*holder, 3*holder), sharey = True)
	for index in range(holder):
		ax[index].boxplot(input[index], showmeans=True, meanline=True, notch=True)
		ax[index].set_title(f"Method: {methods[index]().__class__.__name__}")
	ax[0].set_ylabel('Predicted BMI')
	plt.show()
	if verbose:
		print(input)

@timeit
def visualiseEvaluations(input: np.ndarray,
						 methods: Sequence[Callable],
						 metrics: Sequence[Callable],
						 verbose: bool = False) -> None:
	"""Takes the output of the applyMetrics function and displayes it with matplotlib"""
	holder = len(methods)
	temp = len(metrics)
	fig, ax = plt.subplots(nrows=temp, ncols=1, figsize=(7, 8*holder), sharey = True)
	for i, metric in enumerate(metrics): # Should iterate input col
		for index in range(holder): # Should iterate input row
			ax[i].scatter(index, input[i, index], label = f'{methods[index].__name__}')
			ax[i].set_title(f"Metric: {metrics[i].__name__}")
			ax[i].legend()
			ax[i].grid()
			ax[i].set_ylabel(f'{metrics[i].__name__} Value')
			ax[i].set_xlabel('Method')
	plt.show()
	if verbose:
		print(input)

def main():
	"""Checks that things work"""
	dev = pd.read_csv('../data/development_final_data.csv')
	devData, devMeta = dataSplit(dev)
	train, preds = isolator(devData)

if __name__ == "__main__":
	main()

