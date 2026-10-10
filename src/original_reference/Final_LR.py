import csv
import numpy as np
# import pandas
import math
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from linear_model import LinearModel
from cnn_model import CNNModel

SPLITS = 10

INTRODUCE = False
MINMAX = False
LINDIF = True
DEBUG_DEMO = False
SAVE = False
LOGREG = True 
CNN = False

DEV_SPLIT = 0.05

TIME_LENGTH = 269

REGION_START = 15
REGION_END = -1

TOTAL_FILES = 980
range_files = 980
val_files = TOTAL_FILES - range_files
demo_path = "../demographic.csv"
DATA_PATH = "../ICA_time_course/"
FILE_NAME_ONE = "NCANDA_S0"
FILE_NAME_TWO = "NCANDA_S00"
FILE_NAME_THREE = "NCANDA_S000"
FIRST_FILE = 33
DOT_TXT = ".txt"
NUM_DEMO_FEATURES = 3
THETA_PATH = "./theta_p_features_TV_100.txt"
PRED_PATH = "./pred_p_features_TV_100.txt"
demographics = {}
num_bins = 269

def print_in_lines(arr, n):
	size = len(arr)
	for i in range(int(math.ceil(float(size) / n))):
		if (n * i) + n <= size:
			print(str(n*i) + " - " + str((n * i) + n - 1) + ":" + str(arr[(n*i) : (n*i) + n]))
		else:
			print(str(n*i) + " - " + str(size - 1) + ":" + str(arr[(n*i):]))

def initialize_demographics():
	with open(demo_path, 'tr') as csvfile:
		spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
		counter = 0
		for row in spamreader:
			if counter == 0:
				pass
			else:
				demographics[row[0]] = np.zeros(NUM_DEMO_FEATURES + 1, dtype = float)
				if row[1] == 'M':
					demographics[row[0]][0] = 1 # male = 1; female = -1
				else:
					demographics[row[0]][0] = -1
				demographics[row[0]][1] = row[2] # age
				if row[3] == 'ge':
					demographics[row[0]][2] = 1 # ge = 1; siemens = -1
				else:
					demographics[row[0]][2] = -1
				if row[4] == 'Y':
					demographics[row[0]][3] = 1 # Heavy drinker = 1; Non-heavy drinker = 0
				else:
					demographics[row[0]][3] = 0
			counter += 1

def file_name(i):
	if FIRST_FILE + i >= 1000:
		return FILE_NAME_ONE + str(FIRST_FILE + i)
	elif FIRST_FILE + i >= 100:
		return FILE_NAME_TWO + str(FIRST_FILE + i)
	else:
		return FILE_NAME_THREE + str(FIRST_FILE + i)

def get_lin_features(value, valid = False):
	counter = 0
	lin_dif_info_pop = []
	lin_dif_pop = []
	classifications = []
	if valid:
		ROI = val_files
	else:
		ROI = range_files
	for i in range(ROI):
		FILE_FOUND = True
		f_name = None
		f = None
		try:
			f_name = file_name(i + value)	
			f = np.loadtxt(DATA_PATH + f_name + DOT_TXT)
		except OSError:
			FILE_FOUND = False
		if FILE_FOUND:
			if DEBUG_DEMO:
				if demographics[f_name][3]:
					print("Heavy drinker!", flush = True)
					counter += 1
			time_dur = len(f)
			regions = len(f[1])

			# Data introduction
			if INTRODUCE:
				print("FILE NAME IS: " + f_name)
				print("The normalized data is organized as (time, region): ")
				print(np.shape(f))

			# Intial data processing
			min_max = []
			if MINMAX:
				print("Here are the min max readings of each brain region:")
				for j in range(regions): 
					min_max.append([np.min(f[:, j]), np.max(f[:, j])])
				print_in_lines(min_max, 5)

			lin_dif = []
			lin_dif_info = []
			if LINDIF:
				for j in range(regions):
					lin_dif.append(np.max(f[:, j]) - np.min(f[:, j]))
					lin_dif_info.append(np.max(f[:, j]) - np.min(f[:, j]))
				# print("\nHere are the linear differences: \n")
				# print_in_lines(lin_dif, 5)
			lin_dif_info_pop.append(np.concatenate(([1.], lin_dif_info[:], [demographics[f_name][0], demographics[f_name][2]]))) # Intercept
			lin_dif_pop.append(np.array(lin_dif))
			classifications.append(demographics[f_name][-1])
	if DEBUG_DEMO:
		print(counter)
	lin_dif_info_pop = np.array(lin_dif_info_pop)
	classifications = np.array(classifications)
	return [lin_dif_info_pop, classifications]

class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        m = len(y) # 800 lines of data
        n = len(x[0]) # 1, point, point
        
        def sigmoid(x):
            return 1/(1 + np.exp(-x))

        self.theta = np.zeros(n, dtype = float) # First run, initial
        
        k_debug = 0;        
        counter = 0;

        while True:
            gradient = np.zeros(n, dtype = float)
            hessian = np.zeros((n, n), dtype = float)
            for i in range(m): # Within eachrow
                x_row = x[i]
                TX = np.dot(self.theta, x_row)
                gFunc = sigmoid(TX)
                error = y[i] - gFunc # Calculate the error
                gradient += error * x_row
                hessian += np.outer(x_row, x_row) * gFunc * (1 - gFunc)
            gradient = -gradient/m
            hessian = hessian/m
            hessianInv = np.linalg.inv(hessian)
            thetaD = np.dot(hessianInv, gradient)
            self.theta -= thetaD
            k_debug = np.linalg.norm(thetaD)
            counter += 1
            print("poop " + str(np.log10(k_debug)))
            if np.log10(k_debug) < np.log10(self.eps):
                break;
        return self.theta
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***

        def sigmoid(x):
            return 1/(1 + np.exp(-x))

        XT = np.dot(x, self.theta)
        sFunc = np.vectorize(sigmoid)
        return sFunc(XT)

        # *** END CODE HERE ***

class RNN(CNNModel):
	def fit(self):
		CNN_input = []
		CNN_actual = []
		f_all = []
		for i in range(TOTAL_FILES):
			FILE_FOUND = True
			f_name = None
			f = None
			try:
				f_name = file_name(i)	
				f = np.loadtxt(DATA_PATH + f_name + DOT_TXT)[0 : TIME_LENGTH,:]
				f_all.append(f)
			except OSError:
				FILE_FOUND = False
		print(np.shape(f_all))
		# for i in range(TOTAL_FILES):
		# 	FILE_FOUND = True
		# 	f_name = None
		# 	f = None
		# 	try:
		# 		f_name = file_name(i)	
		# 		f = np.loadtxt(DATA_PATH + f_name + DOT_TXT)[0 : TIME_LENGTH,:]
		# 	except OSError:
		# 		FILE_FOUND = False
		# 	if FILE_FOUND:
		# 		if DEBUG_DEMO:
		# 			if demographics[f_name][3]:
		# 				print("Heavy drinker!", flush = True)
		# 				counter += 1
		# 		time_dur = TIME_LENGTH # n
		# 		regions = len(f[1])
		# 		window = 3 #d
		# 		stride = 1 #s
		# 		m = int(((time_dur - window) / stride) + 1)
		# 		w = np.ones(window, dtype = float)
		# 		z = np.zeros((m,regions), dtype = float)
		# 		for j in range(m):
		# 			z[j] += np.sum(f[j : j + window, :], axis = 0)
		# 		CNN_input.append(z)
		# 		CNN_actual.append(demographics[f_name][-1])
		# m, t, r = np.shape(CNN_input)
		# print(m)
		# print(t)
		# print(r)
		# RNN_model = RNN()
		# RNN_model.fit(CNN_input, CNN_actual, w)

		# for i in range(m):
		# 	patient = CNN_input[i]

def main():
	initialize_demographics()

	if LINDIF:
		x_train_full, y_train_full = get_lin_features(0, False)
		x_val_full, y_val_full = get_lin_features(0, False)

		ICA_T = (y_train_full == 1) ## True at indices where ICA = 1
		ICA_F = (y_train_full == 0)
		# print(ICA_T[24])
		# print(ICA_T[20])
		ICA_1 = y_train_full[ICA_T]
		ICA_0 = y_train_full[ICA_F]
		f_all_1 = []
		f_all_0 = []
		counter = 0
		for trfa in ICA_T:
			if trfa:
				f_all_1.append(x_train_full[counter])
			else:
				f_all_0.append(x_train_full[counter])
			counter += 1 
		trues = len(ICA_1)

		ICA_0 = ICA_0[0:trues]
		f_all_0 = f_all_0[0:trues]

		y_train_full = np.concatenate((ICA_0, ICA_1))
		x_train_full = np.concatenate((f_all_0, f_all_1))
    # *** START CODE HERE ***
	if LOGREG:
		kf = KFold(n_splits = SPLITS, shuffle = True)

		avg_train_accuracy = 0.0
		avg_test_accuracy = 0.0
		avg_dev_accuracy = 0.0
		avg_train_f1_score = 0.0
		avg_test_f1_score = 0.0
		avg_actual_heavy = 0.0
		avg_precision = 0.0
		k_counter = 1

		x_train_full, x_dev, y_train_full, y_dev = train_test_split(x_train_full, y_train_full, random_state = 0, shuffle = True, test_size = DEV_SPLIT)

		for train_index, test_index in kf.split(x_train_full):
			print("\n\n\nKFold step #" + str(k_counter), flush = True)
			x_train, x_test = x_train_full[train_index], x_train_full[test_index]
			y_train, y_test = y_train_full[train_index], y_train_full[test_index]
			x_train = np.array(x_train, dtype = float)
			x_test = np.array(x_test, dtype = float)
			y_train = np.array(y_train, dtype = float)
			y_test = np.array(y_test, dtype = float)

			logReg = LogisticRegression()
			print("beginning fit...", flush = True)
		    # Train a logistic regression classifier
			theta = logReg.fit(x_train, y_train);

		    # Plot decision boundary on top of validation set set
		    #util.plot(x_eval, y_eval, theta, "C:/Users/Joseph/Google Drive/p01b.png")

		    # Use np.savetxt to save predictions on eval set to pred_path

			predictions = logReg.predict(x_test)
			dev_predictions = logReg.predict(x_dev)
			train_predictions = logReg.predict(x_train)
			predictions = (predictions >= 0.5)
			dev_predictions = (dev_predictions >= 0.5)
			train_predictions = (train_predictions >= 0.5)

			recall = np.sum(y_test[(predictions.squeeze() == 1)])/(np.sum(y_test[(predictions.squeeze() == 1)]) + np.sum(y_test[(predictions.squeeze() == 0)]))
			precision = np.sum(y_test[(predictions.squeeze() == 1)])/np.sum(predictions)
			f1_score = 2*precision*recall/(precision+recall)

			train_recall = np.sum(y_train[(train_predictions.squeeze() == 1)])/(np.sum(y_train[(train_predictions.squeeze() == 1)]) + np.sum(y_train[(train_predictions.squeeze() == 0)]))
			train_precision = np.sum(y_train[(train_predictions.squeeze() == 1)])/np.sum(train_predictions)
			train_f1_score = 2*train_precision*train_recall/(train_precision+train_recall)

			print("F1 Score is " + str(f1_score))
			print("Train F1 Score is " + str(train_f1_score))

			avg_train_accuracy += np.sum(train_predictions == y_train) / len(y_train)
			avg_test_accuracy += np.sum(predictions == y_test) / len(y_test)
			avg_dev_accuracy += np.sum(dev_predictions == y_dev) / len(y_dev)
			avg_train_f1_score += train_f1_score
			avg_test_f1_score += f1_score
			avg_actual_heavy += np.sum(y_test) / float(len(y_test))
			avg_precision += precision


			print(str(np.sum(predictions == y_test)) + " is the number correct")
			print("Accuracy then is " + str(np.sum(predictions == y_test) / len(y_test)))
			print(str(np.sum(predictions == 1)) + " was classified as heavy drinkers")
			print(str(np.sum(y_test * (predictions == 1))) + " are correctly classified heavy drinkers")
			print(str(np.sum(y_test)) + " are actually heavy drinkers in the validation set")
			print(str(len(y_test)) + " is the number of ppl in the validation set")
			print(theta)

			if SAVE:
				np.savetxt(PRED_PATH, predictions)
				np.savetxt(THETA_PATH, theta)

			k_counter += 1
		avg_train_accuracy = avg_train_accuracy / SPLITS
		avg_test_accuracy = avg_test_accuracy / SPLITS
		avg_dev_accuracy = avg_dev_accuracy / SPLITS
		avg_train_f1_score = avg_train_f1_score / SPLITS
		avg_test_f1_score = avg_test_f1_score / SPLITS
		avg_actual_heavy = avg_actual_heavy / SPLITS
		avg_precision = avg_precision / SPLITS
		print("For ep value of " + str(logReg.eps) + ", the difference in train/dev accuracy was " + str(avg_train_accuracy - avg_dev_accuracy))
		print("KFOLD RESULTS: \n\n" + "Train accuracy: " + str(avg_train_accuracy) + "\nTrain f1_score: " + str(avg_train_f1_score))
		print("\nTest accuracy: " + str(avg_test_accuracy) + "\nTest f1_score: " + str(avg_test_f1_score))
		print("\nTest actual_heavy: " + str(avg_actual_heavy) + "\nTest precision: " + str(avg_precision))

		
	if CNN:
		RNN_model = RNN()
		RNN_model.fit()


if __name__ == "__main__":
    main()


