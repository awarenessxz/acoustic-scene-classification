"""
simple logging controller that make use of logging module

Referenced from https://oliverleach.wordpress.com/2016/06/15/creating-multiple-log-files-using-python-logging-library/
"""

import logging

"""
	EXAMPLE: 

	import loghub

	loghub.setuplogger('first_logger', 'log1.log')
	loghub.logMsg(__name__, "after main", "first_logger", "info")
"""

formatter = logging.Formatter(fmt="%(asctime)s -> %(levelname)s: %(message)s", datefmt='%m/%d/%Y %I:%M:%S %p')

def init(filename):
	"""
		Function to initialize main log file
	"""
	logging.basicConfig(filename=filename, level=logging.DEBUG, 
		format='%(asctime)s -> %(name)s: %(levelname)s: %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

def setup_logger(logger_name, log_file, level=logging.INFO):
	"""
		Function to setup as many loggers as you want
	"""
	fileHandler = logging.FileHandler(log_file, mode='a')
	fileHandler.setFormatter(formatter)

	streamHandler = logging.StreamHandler()
	streamHandler.setFormatter(formatter)

	logger = logging.getLogger(logger_name)
	logger.setLevel(level)
	logger.addHandler(fileHandler)
	logger.addHandler(streamHandler)

def logMsg(name, msg, otherfile="", level="info"):
	"""
		Log message to main file. You can indicate other file to log the message into that file as well.
	"""
	
	# Log to main file	
	logger = logging.getLogger(name)

	if level == "debug":
		logger.debug(msg)
	elif level == "warning":
		logger.warning(msg)
	elif level == "error":
		logger.error(msg)
	elif level == "critical":
		logger.critical(msg)
	else:
		logger.info(msg)

	# log to other file
	if otherfile:
		logger = logging.getLogger(otherfile)

		if level == "debug":
			logger.debug(msg)
		elif level == "warning":
			logger.warning(msg)
		elif level == "error":
			logger.error(msg)
		elif level == "critical":
			logger.critical(msg)
		else:
			logger.info(msg)







