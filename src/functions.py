import pandas as pd
from collections.abc import Callable
from time import time

def timeit(func: Callable) -> Callable:
	"""Prints the time a function took to execute"""
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
	data = dataframe.iloc[:, 4:]
	metadata = dataframe.iloc[:, 0:4]
	return data, metadata

@timeit
def isolator(data: pd.DataFrame) -> tuple:
	"""Isolates the x's and y's of the dataframe into different variables"""
	x = data.iloc[:, 1:]
	y = data['BMI']
	return x, y

def main():
	"""Checks that things work"""
	dev = pd.read_csv('../data/development_final_data.csv')
	devData, devMeta = dataSplit(dev)
	train, preds = isolator(devData)

if __name__ == "__main__":
	main()

