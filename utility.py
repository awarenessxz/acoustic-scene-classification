"""
Utility functions / Classes 
"""

import datetime
import time

class Namespace:
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






