# You will notice that my variable names inside function tend to be
# different from the ones I use otherwise style wise. I do this because
# the locality of a function allows for reuse of variable names and I
# really like small simple variable names (you can probably tell)

from typing import Sequence
from numpy._typing import _UnknownType
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from collections.abc import Callable
from time import time
from sklearn.utils import resample
from typing import Any
from tqdm import tqdm
from sklearn.model_selection import RepeatedKFold
import joblib
from sklearn.decomposition import PCA

def timeit(func: Callable) -> Callable:
	"""Prints the time a function took to execute"""
	# This is my favorite functions and if you look at my other repositories on github it becomes quite obvious
	def wrapper(*args, showTime: bool = True, **kwargs):
		start= time()
		result = func(*args, **kwargs)
		end = time()
		elapsed = end - start
		if showTime:
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
	x = data.drop(columns = ['BMI']) # everything other than the BMI
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
				[0]
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
		ax[index].boxplot(input[index], showmeans=True, meanline=True, notch=True, sym = '.')
		ax[index].set_title(f"Method: {methods[index]().__class__.__name__}")
		ax[index].grid()
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
	for i in range(temp): # Should iterate input col
		for index in range(holder): # Should iterate input row
			ax[i].scatter(index, input[i, index], label = f'{methods[index].__name__}')
			ax[i].set_title(f"Metric: {metrics[i].__name__}") # Funciton are first class objects in python so __name__ just returns the function name string
			ax[i].legend()
			ax[i].grid()
			ax[i].set_ylabel(f'{metrics[i].__name__} Value')
			ax[i].set_xlabel('Method')
	plt.show()
	if verbose:
		print(input)

@timeit
def baseline(train: pd.DataFrame,
			 labels: pd.DataFrame,
			 val: pd.DataFrame,
			 truth: pd.DataFrame,
			 methods: Sequence[Callable],
			 metrics: Sequence[Callable],
			 plot: bool = False,
			 verbose: bool = False) -> tuple[np.ndarray, np.ndarray]:
	"""Implements the preceding functions to run the entire pipline for the baseline models"""
	predictions = regressions(methods, train, labels, val)
	scores = applyMetrics(metrics, methods, truth, predictions)
	if plot:
		visualisePredictions(predictions, methods, verbose = verbose)
		visualiseEvaluations(scores, methods, metrics, verbose = verbose)
	return predictions, scores

@timeit
def fitting(methods: Sequence[Callable],
			train: pd.DataFrame,
			pred: pd.DataFrame) -> list:
	"""The purpose of this function is to return the model objects form sci kit learn after the fit happens"""
	# I made this function so that I don't re train the models for each loop of the bootstrap
	output = [0] * len(methods)
	for index, model in enumerate(methods):
		object = model()
		object.fit(train, pred)
		output[index] = object
	return output

@timeit
def useFitModels(fitMethods: Sequence[Any], val: pd.DataFrame) -> np.ndarray:
	"""This function pairs with fitting to not have to call the .fit() for each iteration of a bootstrap"""
	# The type hint needs any since callable is a function and the .predict is not a function method
	scale = len(fitMethods)
	# Check the regression funtion for why this data structure
	predictions = np.array(
		[
			np.zeros(
				val.shape[0]
			)
		] * scale
	)
	for index, model in enumerate(fitMethods):
		predictions[index] = model.predict(val)
	return predictions

@timeit
def bootstrapBoth(n: int,
					  dev : pd.DataFrame, # There was a reason I didn't use keyword arguments
					  val: pd.DataFrame, # I can't remember why
					  methods: Sequence[Callable],
					  metrics: Sequence[Callable],
					  plot: bool = False,
					  verbose: bool = False) -> tuple:
	"""Runs the baseline functions multiple times storing outputs and changing both the training and validation set"""
	holder = len(methods)
	predictions = np.array(
		[
			[
				np.zeros(
					val.shape[0]
				)
			] * holder
		] * n
	)
	scores = np.array(
		[
			[
				[
					# See previous function for reasoning behind complexity here
					[0]
				] * holder
			] * len(metrics)
		] * n
	)
	for i in tqdm(range(n)):
		# Permuting data
		train, labels = isolator(resample(dev), showTime = False)
		fitModels = fitting(methods, train, labels, showTime = False)
		permutation = resample(val, random_state = 42)
		validation, true = isolator(permutation, showTime = False)
		predictions[i] = useFitModels(fitModels, validation, showTime = False)
		scores[i] = applyMetrics(metrics, methods, true, predictions[i], showTime = False)
	if plot:
		showConfidence((predictions, scores), methods, metrics, verbose = False)
	if verbose:
		print(predictions[0:3, 0:10, :])
		print(scores[0:3, 0:10, :])
	return predictions, scores

@timeit
def bootstrapVal(n: int,
				 dev : pd.DataFrame, # There was a reason I didn't use keyword arguments
				 val: pd.DataFrame, # I can't remember why
				 methods: Sequence[Callable],
				 metrics: Sequence[Callable],
				 plot: bool = False,
				 verbose: bool = False) -> tuple:
	"""Runs the baseline functions multiple times storing outputs and changing the validation set"""
	holder = len(methods)
	predictions = np.array(
		[
			[
				np.zeros(
					val.shape[0]
				)
			] * holder
		] * n
	)
	scores = np.array(
		[
			[
				[
					# See previous function for reasoning behind complexity here
					[0]
				] * holder
			] * len(metrics)
		] * n
	)
	train, labels = isolator(dev, showTime = False)
	fitModels = fitting(methods, train, labels, showTime = False)
	for i in tqdm(range(n)):
		# Permuting data
		permutation = resample(val, random_state = 42)
		validation, true = isolator(permutation, showTime = False)
		predictions[i] = useFitModels(fitModels, validation, showTime = False)
		scores[i] = applyMetrics(metrics, methods, true, predictions[i], showTime = False)
	if plot:
		showConfidence((predictions, scores), methods, metrics, verbose = False)
	if verbose:
		print(predictions[0:3, 0:10, :])
		print(scores[0:3, 0:10, :])
	return predictions, scores

@timeit
def bootstrapDev(n: int,
				  dev : pd.DataFrame, # There was a reason I didn't use keyword arguments
				  val: pd.DataFrame, # I can't remember why
				  methods: Sequence[Callable],
				  metrics: Sequence[Callable],
				  plot: bool = False,
				  verbose: bool = False) -> tuple:
	"""Runs the baseline functions multiple times storing outputs and changing the development set"""
	holder = len(methods)
	predictions = np.array(
		[
			[
				np.zeros(
					val.shape[0]
				)
			] * holder
		] * n
	)
	scores = np.array(
		[
			[
				[
					# See previous function for reasoning behind complexity here
					[0]
				] * holder
			] * len(metrics)
		] * n
	)
	validation, true = isolator(val, showTime = False)
	for i in tqdm(range(n)):
		# Permuting data
		train, labels = isolator(resample(dev), showTime = False)
		fitModels = fitting(methods, train, labels, showTime = False)
		predictions[i] = useFitModels(fitModels, validation, showTime = False)
		scores[i] = applyMetrics(metrics, methods, true, predictions[i], showTime = False)
	if plot:
		showConfidence((predictions, scores), methods, metrics, verbose = False)
	if verbose:
		print(predictions[0:3, 0:10, :])
		print(scores[0:3, 0:10, :])
	return predictions, scores

@timeit
def showConfidence(preds: tuple,
				   methods: Sequence[Callable],
				   metrics: Sequence[Callable],
				   verbose: bool = False) -> None:
	"""Takes the output of bootstrapBaseline and plots it"""
	predictions = preds[0]
	scores = preds[1]
	holder = len(methods)
	# Predictions segment
	fig, ax = plt.subplots(nrows=1, ncols=holder, figsize=(5*holder, 7*holder), sharey = True)
	for index in range(holder):
		ax[index].boxplot(predictions[:, index, :], showmeans=True, meanline=True, notch = True, sym = '.')
		ax[index].set_title(f"Method: {methods[index]().__class__.__name__}")
		ax[index].grid(axis = 'y')
	ax[0].set_ylabel('Predicted BMI')
	plt.show()
	if verbose:
		print(predictions)
	# Scores segment
	temp = len(metrics)
	fig, ax = plt.subplots(nrows=holder, ncols=temp, figsize=(6*holder, 4*temp), sharey = True)
	for i in range(temp): # Should iterate input col
		for index in range(holder): # Should iterate input row
			ax[index, i].boxplot(scores[:, i, index, :], showmeans=True, meanline=True, sym = '.') # index, i is row, col in matplotlib
			ax[index, i].set_title(f"Metric: {metrics[i].__name__}") # Funciton are first class objects in python so __name__ just returns the function name string
			ax[index, i].grid()
			ax[index, i].set_ylabel(f'{metrics[i].__name__} Value')
			ax[index, i].set_xlabel(f'Method: {methods[index].__name__}')
	plt.show()
	if verbose:
		print(scores)

@timeit
def applyKfold(train: pd.DataFrame,
			   labels: pd.DataFrame,
			   methods: Sequence[Callable],
			   metrics: Sequence[Callable],
			   splits: int,
			   iterations: int,
			   plot: bool = True,
			   verbose: bool = False) -> np.ndarray:
	"""This function takes the baseline and applies a k fold cross validation protocol to it"""
	holder = len(methods)
	# I want a data structure that can hold the output of apply Metrics for each fold
	scores = np.array(
		[
			[
				[
					# See previous function for reasoning behind complexity here
					[0]
				] * holder
			] * len(metrics)
		] * splits * iterations
	)
	kf = RepeatedKFold(n_splits= splits, n_repeats=iterations, random_state = 42) # I rely on the documentation heavily here
	# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RepeatedKFold.html#sklearn.model_selection.RepeatedKFold
	for index, holder in tqdm(enumerate(kf.split(train))):
		x = train.iloc[holder[0]]
		val = train.iloc[holder[1]]
		preds = labels.iloc[holder[0]]
		models = fitting(methods, x, preds, showTime = False)
		results = useFitModels(models, val, showTime = False)
		scores[index - 1] = applyMetrics(metrics, methods, labels.iloc[holder[1]], results, showTime = False)
	if plot:
		holder = len(methods)
		temp = len(metrics)
		fig, ax = plt.subplots(nrows=holder, ncols=temp, figsize=(6*holder, 4*temp), sharey = True)
		for i in range(temp): # Should iterate input col
			for index in range(holder): # Should iterate input row
				ax[index, i].boxplot(scores[:, i, index, :], showmeans=True, meanline=True, sym = '.') # index, i is row, col in matplotlib
				ax[index, i].set_title(f"Metric: {metrics[i].__name__}") # Funciton are first class objects in python so __name__ just returns the function name string
				ax[index, i].grid()
				ax[index, i].set_ylabel(f'{metrics[i].__name__} Value')
				ax[index, i].set_xlabel(f'Method: {methods[index].__name__}')
		plt.show()
	if verbose:
		print(scores[0:3, 0:3, 0:3, 0:3])
	return scores

@timeit
def saveModels(models: list, folder: os.PathLike, identifier: str) -> None:
	"""Saves all models in folder with their name and a unique identifier"""
	os.makedirs(folder, exist_ok = True)
	for model in models:
		joblib.dump(model, f"{folder}{model.__class__.__name__}_{identifier}.pkl")

@timeit
def applyPca(data: pd.DataFrame, plot: bool = True) -> PCA:
	"""Plots the explain values for each feature"""
	# I notmalize the blue curve to the red one so that both are visible in detail
	pca = PCA()
	pca.fit(data)
	if plot:
		plt.plot(range(1, len(data.columns) + 1), pca.explained_variance_ratio_, color = 'red', label = 'Raw variance')
		plt.plot(range(1, len(data.columns) + 1), [0.06 * sum(pca.explained_variance_ratio_[0:i]) for i in range(len(data.columns))], color = 'blue', label = 'Cummulative variance')
		plt.grid()
		plt.legend()
		plt.title('Explain (%) of principal components')
		plt.xlabel('Principal component')
		plt.ylabel('Explained variance')
		plt.show()
	return pca.transform(data)

@timeit
def fname(arg: type) -> None:
	"""doc"""
	# body
	return None

def main():
	"""Checks that things work"""
	pass

if __name__ == "__main__":
	main()

