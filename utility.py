"""
Utility functions / Classes 
"""

import datetime
import time

'''
////////////////////////////////////////////////////////////////////////////////////
///					General Function											////
////////////////////////////////////////////////////////////////////////////////////
'''

def isStrEmpty(input_str):
	if not input_str:
		print("Empty")
	else:
		print("Not Empty")

def isListEmpty(input_list):
	if not input_list:
		print("Empty List")
	else:
		print("Not Empty")

def compare_list_elements(listA, listB):
	"""
		Compare two list element wise and return number of correct elements
		Note: List should be same size
	"""

	sizeA = len(listA)
	sizeB = len(listB)

	if sizeA != sizeB:
		print("ERROR LIST ARE NOT EQUAL IN SIZE")
		return

	total = sizeA
	correct = 0

	for index, (x,y) in enumerate(zip(listA, listB)):
		if x == y:
			correct += 1

	return correct, total


'''
////////////////////////////////////////////////////////////////////////////////////
///					Useful Classes												////
////////////////////////////////////////////////////////////////////////////////////
'''

class Namespace:
	"""
		create your own dictionary of args
		convert that dictionary into format similar to args = argparse.ArgumentParser().parse_args()
		
		Eg. 
		args = {
			"batch_size": 16,
			"no_cuda": True,
			"seed": 1,
		}
		args = Namespace(**args)
		
		print(args.batch_size)
	"""

	def __init__(self, **kwargs):
		self.__dict__.update(kwargs)

class StopWatch:
	"""
		Class for running program time
	"""

	def __init__(self):
		self.start = None
		self.end = None

	def resetTimer(self):
		self.start = None
		self.end = None

	def startTimer(self):
		"""
			Start timing
		"""
		self.resetTimer()
		self.start = time.time()

	def stopTimer(self):
		"""
			Stop timing
		"""

		# ensures that timer have started
		if not self.start:
			print("Unable to stop timer. No start time found. Please run 'startTimer()' first...")
		else:
			self.end = time.time()

	def printElapsedTime(self):
		"""
			Returns the time taken since stopwatch started.
		"""
		if not self.start:
			print("No start time found. Please run 'startTimer()' first...")
		elif not self.end:
			current_time = time.time() - self.start
			print("Current run time: %f. Stop watch is still running...Run 'stopTimer()' to stop the timer" % (current_time))
		else:
			time_taken = self.end - self.start
			print("Total time taken in seconds: %f s" % time_taken)
			print("Total time taken: %s" % str(datetime.timedelta(seconds=time_taken)))

	def getElapsedTime(self):
		time_taken = self.end - self.start
		return str(datetime.timedelta(seconds=time_taken))






