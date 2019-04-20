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
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

import pickle

'''
////////////////////////////////////////////////////////////////////////////////////
///					General Function											////
////////////////////////////////////////////////////////////////////////////////////
'''

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

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

	def __init__(self, x_train, y_train, x_test, y_test):
		self.x_train = x_train
		self.y_train = y_train
		self.x_test = x_test
		self.y_test = y_test
		self.model = None

	'''
	Train and Predict using Nayes Bayes Classification Model
	'''
	def run_nayes_bayes_classification(self):
		self.model = MultinomialNB().fit(self.x_train, self.y_train)
		predicts = self.model.predict(self.x_test)

		return predicts

	'''
	Train and Predict using K Nearest Neighbour Classification Model
	'''
	def run_KNN_classification(self, k=11):
		self.model = KNeighborsClassifier(n_neighbors=k)
		self.model.fit(self.x_train, self.y_train)
		predicts = self.model.predict(self.x_test)

		return predicts

	'''
	Train and Predict using Decision Tree Classification Model
	'''
	def run_decision_tree_classification(self):
		self.model = DecisionTreeClassifier()
		self.model.fit(self.x_train, self.y_train)
		predicts = self.model.predict(self.x_test)

		return predicts

	'''
	Train and Predict using Random Forest Classification Model
	'''
	def run_random_forest_classification(self):
		self.model = RandomForestClassifier(max_depth=5, n_estimators=10)
		self.model.fit(self.x_train, self.y_train)
		predicts = self.model.predict(self.x_test)

		feature_importances = self.model.feature_importances_

		return predicts, feature_importances

	'''
	Train and Predict using Support Vector Machine (SVM) Classification Model
	'''
	def run_SVM_classification(self):
		self.model = LinearSVC(random_state=0, tol=1e-5)
		self.model.fit(self.x_train, self.y_train)
		predicts = self.model.predict(self.x_test)

		return predicts

	'''
	Train and Predict using Stochastic Gradient Descent (SGD) Classification Model
	'''
	def run_SGD_classification(self):
		self.model = SGDClassifier(max_iter=1000, tol=1e-3)
		self.model.fit(self.x_train, self.y_train)
		predicts = self.model.predict(self.x_test)

		return predicts

	'''
	Train and Predict using Logistic Regression Classification Model
	'''
	def run_LR_classification(self):
		self.model = LogisticRegression()
		self.model.fit(self.x_train, self.y_train)
		predicts = self.model.predict(self.x_test)

		return predicts

	'''
	Evaluate prediction based on precision, recall, f_meaure
	'''
	def evaluate_prediction(self, predicts):
		#print(classification_report(self.y_test, self.predicts))
		precision = precision_score(self.y_test, predicts, average='macro')
		recall = recall_score(self.y_test, predicts, average='macro')

		f_score = compute_F_measure(precision, recall)

		return precision, recall, f_score

	'''
	Get Accuracy 
	'''
	def get_accuracy(self, predicts):
		total = len(predicts)
		correct = 0

		for index, (x, y) in enumerate(zip(predicts, self.y_test)):
			if x == y:
				correct += 1

		return correct, total

	'''
	Save Model
	'''
	def save_model(self, filename):
		pickle.dump(self.model, open(filename, 'wb'))

	'''
	Load Model
	'''
	def load_model(self, filename):
		self.model = pickle.load(open(filename, 'rb'))
