# You will notice that my variable names inside function tend to be
# different from the ones I use otherwise style wise. I do this because
# the locality of a function allows for reuse of variable names and I
# really like small simple variable names (you can probably tell)

from typing import Sequence
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from collections.abc import Callable
from time import time
from sklearn.linear_model import BayesianRidge, ElasticNet
from sklearn.svm import SVR
from sklearn.utils import resample
from typing import Any
from tqdm import tqdm
from sklearn.model_selection import RepeatedKFold
import joblib
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import optuna


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
				val: pd.DataFrame,
				arguments: list | None = None) -> np.ndarray:
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
	if arguments:
		index = 0
		for model, args in zip(methods, arguments):
			object = model(**args)
			object.fit(train,pred)
			predictions[index] = object.predict(val)
			index +=1
	else:
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
				[0] # Ill use this third dimension for the bootstrap later don't mind it now
			] * holder
		] * len(metrics)
	)
	# So we will end up with the following
	# [[[option1], [option2], [option3], ...], # metric1
	# [[option1], [option2], [option3], ...], # metric2
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
			pred: pd.DataFrame,
			arguments: list | None = None) -> list:
	"""The purpose of this function is to return the model objects form sci kit learn after the fit happens"""
	# I made this function so that I don't re train the models for each loop of the bootstrap
	output = [0] * len(methods)
	if arguments:
		index = 0
		for model, args in zip(methods, arguments):
			object = model(**args)
			object.fit(train,pred)
			output[index] = object
			index += 1
		del index
	else:
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
				  arguments: list | None = None,
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
		fitModels = fitting(methods, train, labels, arguments = arguments, showTime = False)
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
				 arguments: list | None = None,
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
	fitModels = fitting(methods, train, labels, arguments = arguments, showTime = False)
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
				  arguments: list | None = None,
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
		fitModels = fitting(methods, train, labels, arguments = arguments, showTime = False)
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
			   arguments: list | None = None,
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
		models = fitting(methods, x, preds, arguments = arguments, showTime = False)
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
def searchPca(data: pd.DataFrame) -> None:
	"""Calculates and plots the explain values for each feature"""
	# I notmalize the blue curve to the red one so that both are visible in detail
	pca = PCA()
	pca.fit(data)
	plt.plot(range(1, len(data.columns) + 1), pca.explained_variance_ratio_, color = 'red', label = 'Raw variance')
	plt.plot(range(1, len(data.columns) + 1), [0.06 * sum(pca.explained_variance_ratio_[0:i]) for i in range(len(data.columns))], color = 'blue', label = 'Cummulative variance')
	plt.grid()
	plt.legend()
	plt.title('Explain (%) of principal components')
	plt.xlabel('Principal component')
	plt.ylabel('Explained variance')
	plt.show()

@timeit
def fitPca(data: pd.DataFrame, n: int | float) -> PCA:
	"""Fits the pca and returns the object"""
	return PCA(n_components = n).fit(data)

@timeit
def applyPca(data: pd.DataFrame, pca: PCA) -> pd.DataFrame:
	"""Returns the dataframe with the transformation to n components"""
	return pd.DataFrame(pca.fit_transform(data))

@timeit
def upperTriangle(matrix: np.ndarray) -> tuple[list, list]:
	"""This function is to return the two lists of indecies to use to plot a matrix"""
	# I need two arrays, one for redundant values and one for non redundant ones
	# Since i need something like
	# 1 - 1
	# 2 - 1
	# 2 - 2
	# 3 - 1
	# 3 - 2
	# ...
	width = len(matrix) - 1
	height = width
	volume = int((width) * (height) / 2) # of a triangle (technically an area)
	redundant = [0] * volume
	serial = [0] * volume
	holder = 0
	for i in range(1, len(matrix) - 1): # This one starts at 1 since the 0 index is on the diagonal I guess we could leave it and it would skip it
		for j in range(0, len(matrix) - 1):
			if i > j:
				holder += 1
				redundant[holder] = i
				serial[holder] = j
			else:
				continue
			# Note
			# There is a way to make this faster I know of, I simply dont know if I'll go through the trouble of implementing it that way
			# since this is already fast enough so this note is in case I don't. If i allocated all the memory in the list's before hand and 
			# simply assign i and j instead of appending it would be faster. The syntax would look like redundant = [0]* volume were volume wold need to be
			# the number of elements in the final redundant list.
			#
			# I just got done implementing it, it's says the same time but it might still be faster (just not by alot).
	return redundant, serial

@timeit
def correlation(dataframe: pd.DataFrame, plot: bool = True) -> pd.DataFrame:
	"""This function will take the correlation matrix of the dataframe and make a scatter plot for it (col 1 index, col 2 index, corr)"""
	table = dataframe.corr() # There used to be an iloc here in the other notebook here all features are number however
	# We only need everything above the diagonal
	rows, cols = upperTriangle(table.values)  # Originally solved inside this function now moved to the one above
	if plot:
		fig = plt.figure()
		ax = plt.axes(projection = '3d')
		# I'm going to multiply the corr() output by alot since I want to have the delta z visible on the plot outside of just the color heatmap
		ax.scatter(rows, cols, 1000 * table.values[rows, cols], label = 'Pearson corr', c = table.values[rows, cols], cmap = 'YlOrBr')
		plt.title(f"Correlations of the dataframe")
		ax.set_xlabel(f"Col 1")
		ax.set_ylabel(f"Col 2")
		ax.set_zlabel("Pearson correlation of dataframe")
		plt.grid()
		plt.show()
	return table

@timeit
def pearsonPrune(data: pd.DataFrame, cutoff: float | int) -> pd.DataFrame:
	"""Removes entries in the dataframe that have correlation higher than cutoff"""
	# I'm goingto removve the features from the end of the table since if I do it the other way
	# I can run into an indexing error.
	table = data.corr()
	rows, cols = upperTriangle(table.values)
	for row in rows:
		for col in cols:
			if table.values[row, col] >= cutoff:
				data = data.drop(index = col, axis = 1)
	return data

@timeit
def compareMetrics(data: np.ndarray, compare: np.ndarray) -> np.ndarray:
	"""Takes the output of applyMetrics for two runs and outputs the comparison for all metrics just numerically"""
	shape = data.shape
	# Making an output that is the same as the metric lists
	# Variable to keep the metrics
	output = data.copy()
	for i in range(shape[0]): # metrics
		for j in range(shape[1]): # methods
			output[i, j, 0] = data[i, j, 0] - compare[i, j, 0]
	return output

@timeit
def compareBootstrap(data: np.ndarray, compare: np.ndarray) -> tuple:
	"""Takes the output of bootstrap for two runs and outputs the comparison for all metrics just numerically"""
	# One can imagine an error because the two bootstraps have different sizes. I won't be fixing that.
	dataShape = data[0].shape # (n, methods)
	scoresShape = data[1].shape # (n, metrics, methods)
	dataOutput = data[0].copy()
	scoresOutput = data[1].copy()
	# data section
	# I don't really need this one but all the architecture is here so why not
	for i in range(dataShape[0]): # n
		for j in range(dataShape[1]): # methods
			dataOutput[i, j, 0] = data[i, j, 0] - compare[i, j, 0]
	# Scores section
	for i in range(scoresShape[0]): # n
		for j in range(scoresShape[1]): # metrics
			scoresOutput[i, j, 0] = data[i, j, 0] - compare[i, j, 0]
	return dataOutput, scoresOutput

@timeit
def compareKfold(data: np.ndarray, compare: np.ndarray) -> np.ndarray:
	"""Takes the output of bootstrap for two runs and outputs the comparison for all metrics just numerically"""
	# One can imagine an error because the two bootstraps have different sizes. I won't be fixing that.
	shape = data.shape # (splits * iterations, metrics, methods)
	output = data.copy()
	for i in range(shape[0]): # splits * iterations
		for j in range(shape[1]): # metrics
			for k in range(shape[2]): # methods
				output[i, j, k, 0] = data[i, j, k, 0] - compare[i, j, k, 0]
	return output

@timeit
def visualiseComparedBootstrap(preds: tuple,
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
		ax[index].boxplot(predictions[:, index, :], showmeans=True, meanline=True, sym = '.')
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
			ax[index, i].boxplot(scores[i, index, :], showmeans=True, meanline=True, sym = '.') # index, i is row, col in matplotlib
			ax[index, i].set_title(f"Metric: {metrics[i].__name__}") # Funciton are first class objects in python so __name__ just returns the function name string
			ax[index, i].grid()
			ax[index, i].set_ylabel(f'{metrics[i].__name__} Value')
			ax[index, i].set_xlabel(f'Method: {methods[index].__name__}')
	plt.show()
	if verbose:
		print(scores)

@timeit
def visualiseComparedKfold(data: np.ndarray, methods: Sequence[Callable], metrics: Sequence[Callable]) -> None:
	"""Plots the compareKfold function output"""
	holder = len(methods)
	temp = len(metrics)
	fig, ax = plt.subplots(nrows=holder, ncols=temp, figsize=(6*holder, 4*temp), sharey = True)
	for i in range(temp): # Should iterate input col
		for index in range(holder): # Should iterate input row
			ax[index, i].boxplot(data[:, i, index, :], showmeans=True, meanline=True, sym = '.') # index, i is row, col in matplotlib
			ax[index, i].set_title(f"Metric: {metrics[i].__name__}") # Funciton are first class objects in python so __name__ just returns the function name string
			ax[index, i].grid()
			ax[index, i].set_ylabel(f'{metrics[i].__name__} Value')
			ax[index, i].set_xlabel(f'Method: {methods[index].__name__}')
	plt.show()

@timeit
def optimizeSVR(metric: Callable,
				train: pd.DataFrame,
				preds: pd.DataFrame,
				val: pd.DataFrame,
				truth: pd.DataFrame,
				trials: int = 100) -> dict:
	"""Returns the constructed SVR class instance after initializing it with the hyperparameter tuning"""
	def objective(trial: optuna.trial.Trial,
			   metric: Callable = metric,
			   train: pd.DataFrame = train,
			   preds: pd.DataFrame = preds,
			   val: pd.DataFrame = val,
			   truth: pd.DataFrame = truth) -> float:
		"""Defines an objective to optimize"""
		c = trial.suggest_float('C', 0.1, 10.0)
		gamma = trial.suggest_categorical('gamma', ['auto', 'scale'])
		epsilon = trial.suggest_float('epsilon', 0.01, 1.0)
		kernel = trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf'])
		degree = trial.suggest_int('degree', 2, 5)
		coef = trial.suggest_float('coef0', 0.0, 1.0)
		model = SVR(C=c, kernel=kernel, degree=degree, coef0=coef, gamma=gamma, epsilon=epsilon)
		model.fit(train, preds)
		output = model.predict(val)
		return metric(output, truth)
	study = optuna.create_study()
	study.optimize(objective, n_trials = trials)
	return study.best_params

@timeit
def optimizeRidge(metric: Callable,
				train: pd.DataFrame,
				preds: pd.DataFrame,
				val: pd.DataFrame,
				truth: pd.DataFrame,
				trials: int = 100) -> dict:
	"""Returns the constructed BayesianRidge class instance after initializing it with the hyperparameter tuning"""
	def objective(trial: optuna.trial.Trial,
			   metric: Callable = metric,
			   train: pd.DataFrame = train,
			   preds: pd.DataFrame = preds,
			   val: pd.DataFrame = val,
			   truth: pd.DataFrame = truth) -> float:
		"""Defines an objective to optimize"""
		alpha1 = trial.suggest_float('alpha_1', 1e-08, 1e-04)
		alpha2 = trial.suggest_float('alpha_2', 1e-08, 1e-04) 
		l1 = trial.suggest_float('lambda_1', 1e-08, 1e-04) 
		l2 = trial.suggest_float('lambda_2', 1e-08, 1e-04) 
		model = BayesianRidge(alpha_1=alpha1, alpha_2=alpha2, lambda_1=l1, lambda_2=l2)
		model.fit(train, preds)
		output = model.predict(val)
		return metric(output, truth)
	study = optuna.create_study()
	study.optimize(objective, n_trials = trials)
	return study.best_params

@timeit
def optimizeNet(metric: Callable,
				train: pd.DataFrame,
				preds: pd.DataFrame,
				val: pd.DataFrame,
				truth: pd.DataFrame,
				trials: int = 100) -> dict:
	"""Returns the constructed ElasticNet class instance after initializing it with the hyperparameter tuning"""
	def objective(trial: optuna.trial.Trial,
			   metric: Callable = metric,
			   train: pd.DataFrame = train,
			   preds: pd.DataFrame = preds,
			   val: pd.DataFrame = val,
			   truth: pd.DataFrame = truth) -> float:
		"""Defines an objective to optimize"""
		alpha = trial.suggest_float('alpha', 0.1, 10.0)
		l1 = trial.suggest_float('l1_ratio', 0.0, 1.0)
		model = ElasticNet(alpha = alpha, l1_ratio = l1, random_state  = 42)
		model.fit(train, preds)
		output = model.predict(val)
		return metric(output, truth)
	study = optuna.create_study()
	study.optimize(objective, n_trials = trials)
	return study.best_params

def main():
	"""Checks that things work"""
	pass

if __name__ == "__main__":
	main()

