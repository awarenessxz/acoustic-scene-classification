'''
General Purpose: Run the machine learning classifier algorithms and return prediction results
'''

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, precision_score, recall_score


'''
////////////////////////////////////////////////////////////////////////////////////
///					General Function											////
////////////////////////////////////////////////////////////////////////////////////
'''

'''
Compute F measure score
'''
def compute_F_measure(p, r, beta=1):
	# prevent division by zero error
	if (p == 0 and r == 0):
		return 0

	return ((pow(beta, 2) + 1) * p * r) / (pow(beta, 2) * p + r)


'''
////////////////////////////////////////////////////////////////////////////////////
///			Class to run the different classifier model							////
////////////////////////////////////////////////////////////////////////////////////
'''

class ClassifierModel:

	def __init__(self, x_train, y_train, x_test):
		self.x_train = x_train
		self.y_train = y_train
		self.x_test = x_test

	'''
	Train and Predict using Nayes Bayes Classification Model
	'''
	def run_nayes_bayes_classification(self):
		model = MultinomialNB().fit(self.x_train, self.y_train)
		predicts = model.predict(self.x_test)

		return predicts

	'''
	Train and Predict using K Nearest Neighbour Classification Model
	'''
	def run_KNN_classification(self, k=11):
		model = KNeighborsClassifier(n_neighbors=k)
		model.fit(self.x_train, self.y_train)
		predicts = model.predict(self.x_test)

		return predicts

	'''
	Train and Predict using Decision Tree Classification Model
	'''
	def run_decision_tree_classification(self):
		model = DecisionTreeClassifier()
		model.fit(self.x_train, self.y_train)
		predicts = model.predict(self.x_test)

		return predicts

	'''
	Train and Predict using Random Forest Classification Model
	'''
	def run_random_forest_classification(self):
		model = RandomForestClassifier(max_depth=5, n_estimators=10)
		model.fit(self.x_train, self.y_train)
		predicts = model.predict(self.x_test)

		feature_importances = model.feature_importances_

		return predicts, feature_importances

	'''
	Train and Predict using Support Vector Machine (SVM) Classification Model
	'''
	def run_SVM_classification(self):
		model = LinearSVC(random_state=0, tol=1e-5)	
		model.fit(self.x_train, self.y_train)
		predicts = model.predict(self.x_test)

		return predicts

	'''
	Train and Predict using Stochastic Gradient Descent (SGD) Classification Model
	'''
	def run_SGD_classification(self):
		model = SGDClassifier(max_iter=1000, tol=1e-3)	
		model.fit(self.x_train, self.y_train)
		predicts = model.predict(self.x_test)

		return predicts

	'''
	Train and Predict using Logistic Regression Classification Model
	'''
	def run_LR_classification(self):
		model = LogisticRegression()
		model.fit(self.x_train, self.y_train)
		predicts = model.predict(self.x_test)

		return predicts

'''
////////////////////////////////////////////////////////////////////////////////////
///			Class to evaluate the prediction results							////
////////////////////////////////////////////////////////////////////////////////////
'''

class PredictionEvaluator:

	def __init__(self, predicts, y_test):
		self.predicts = predicts
		self.y_test = y_test

	'''
	Evaluate prediction based on precision, recall, f_meaure
	'''
	def evaluate_prediction(self):
		#print(classification_report(self.y_test, self.predicts))
		precision = precision_score(self.y_test, self.predicts, average='macro')
		recall = recall_score(self.y_test, self.predicts, average='macro')

		f_score = compute_F_measure(precision, recall)

		return precision, recall, f_score

	def get_accuracy(self):
		





