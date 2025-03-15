# You will notice that my variable names inside function tend to be
# different from the ones I use otherwise style wise. I do this because
# the locality of a function allows for reuse of variable names and I
# really like small simple variable names (you can probably tell)

from typing import Sequence
import pandas as pd
import numpy as np
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
def regression(method: Callable, train: pd.DataFrame, pred: pd.DataFrame, val: pd.DataFrame) -> np.ndarray:
	"""Calls the method on the data given and return outputs"""
	object = method()
	object.fit(train, pred)
	return object.predict(val)

@timeit
def applyMetrics(metrics: Sequence[Callable], truth: np.ndarray, data: np.ndarray) -> np.ndarray:
	"""Applies all the given metrics to the data assuming as ground truth the truth given"""
	# Note that the function takes input as numpy array and not a dataframe which is why we convert the input in the notebook in the ()'s of the function
	# Holder of the outputs
	output = np.array([0] * len(metrics))
	for i, metric in enumerate(metrics):
		output[i] = metric(truth, data)
	return output

def main():
	"""Checks that things work"""
	dev = pd.read_csv('../data/development_final_data.csv')
	devData, devMeta = dataSplit(dev)
	train, preds = isolator(devData)

if __name__ == "__main__":
	main()

