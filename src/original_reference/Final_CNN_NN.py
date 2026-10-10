import os

os.environ["MKL_THREADING_LAYER"] = "GNU"

import keras
import keras.backend as K
from keras.models import Sequential
from keras.layers import Activation, Concatenate, Flatten, TimeDistributed

from sklearn.model_selection import train_test_split, KFold

from theano.tensor import _shared
import numpy as np
# fix random seed for reproducibility


import glob
import csv

import math

#constants#
#Control
TWO_D_CONV = False
CLASS_BALANCE = True
KFOLD = True
CLASS_BALANCE_ADDITION = False
MALE_ONLY = False
FEMALE_ONLY = False
GE_ONLY = False
SIEMENS_ONLY = False
INCLUDE_AGE = False
ADDITIONAL = False
RANDOMS = 10
SPLITS = 10

#Data related
TIME_LENGTH = 269
TEST_FRACTION = 0.01
DEV_SPLIT = 0.05

#Demographics
DEMO_PATH = "time_course/demographic.csv"
NUM_DEMO_FEATURES = 3
DEMOGRAPHICS = {}

## File extraction
DATA_PATH = "time_course/ICA_time_course/" ###'../craddock100_time_course/' ####
PATTERN_MATCH = '/*.txt'
ICA = glob.glob(DATA_PATH + PATTERN_MATCH)

#CNN related
NUM_OUTPUT_FILTERS = 5 # May want to increase this
KERNEL_SIZE = 5 # May want to increase this #####
TWO_D_KERNEL_SIZE = (5, 3)
STRIDE = 1
TWO_D_STRIDE = (1, 1)
DATA_FORMAT = 'channels_last'
CNN_ACTIVATION = 'tanh' #Watch for exploding gradients, dying neurons with relu #####
AGE_ACTIVATION = None

#RNN related
OUTPUT_DIM = 5
RNN_ACTIVATION = 'hard_sigmoid' #####
ADDITIONAL_LSTM = 0
DROPOUT = 0.8
RECURRENT_DROPOUT = 0.5

#Model related
OPTIMIZER = 'adam' #consider different optimizer
LOSS = 'binary_crossentropy' #consider different loss
METRICS = ['accuracy'] #DEFINITELY consider different metrics
EPOCHS = 200
FINAL_ACTIVATION = 'sigmoid'
BATCH_SIZE = 220
ACC_THRESHOLD = 0.6
MIN_ACC = 0.55
MAX_ACC = 0.65
#constants

def initialize_demographics():
	with open(DEMO_PATH, 'rt') as csvfile:
		spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
		counter = 0
		for row in spamreader:
			if counter == 0:
				pass
			else:
				DEMOGRAPHICS[row[0]] = np.zeros(NUM_DEMO_FEATURES + 1, dtype = float)
				if row[1] == 'M':
					DEMOGRAPHICS[row[0]][0] = 1 # male = 1; female = -1
				else:
					DEMOGRAPHICS[row[0]][0] = -1
				DEMOGRAPHICS[row[0]][1] = row[2]
				if row[3] == 'ge':
					DEMOGRAPHICS[row[0]][2] = 1 # ge = 1; siemens = -1
				else:
					DEMOGRAPHICS[row[0]][2] = -1
				if row[4] == 'Y':
					DEMOGRAPHICS[row[0]][3] = 1 # Heavy drinker = 1; Non-heavy drinker = 0
				else:
					DEMOGRAPHICS[row[0]][3] = 0
			counter += 1
def process_demographics(f_all, cICA, age_all):
	new_f_all = []
	new_cICA = []
	new_age_all = []
	counter = 0
	for name in ICA:
		f_name = name[-17:-4]
		if MALE_ONLY and DEMOGRAPHICS[f_name][0] == -1:
			counter +=1
			continue
		if FEMALE_ONLY and DEMOGRAPHICS[f_name][0] == 1:
			counter +=1
			continue
		if GE_ONLY and DEMOGRAPHICS[f_name][2] == -1:
			counter +=1 
			continue
		if SIEMENS_ONLY and DEMOGRAPHICS[f_name][2] == 1:
			counter +=1
			continue
		new_f_all.append(f_all[counter])
		new_cICA.append(cICA[counter])
		new_age_all.append(age_all[counter])
		counter +=1
	return np.array(new_f_all), np.array(new_cICA), np.array(new_age_all)
def main_sub(seed):
	initialize_demographics()
	CNN_input = []
	CNN_actual = []
	f_all = []
	age_all = []
	counter = 0
	for f_name in ICA:
		if f_name != ICA[counter]:
			print("ORDERING OF DEMOGRAPHICS AND FILES IS NOT THE SAME")
		counter += 1
		FILE_FOUND = True
		f = None
		try:	
			f = np.loadtxt(f_name)[0 : TIME_LENGTH,:]
		except OSError:
			FILE_FOUND = False
			print("ERROR: A FILE WAS NOT OPENED PROPERLY!!!")
		f_all.append(f)
		age_all.append(DEMOGRAPHICS[f_name[-17:-4]][1])

	def preprocess_y(a):
		return DEMOGRAPHICS[a[-17:-4]][3] #Gets just the file name without path & .txt

	yf = np.vectorize(preprocess_y)
	cICA = np.array(yf(ICA))

	f_all, cICA, age_all = process_demographics(f_all, cICA, age_all)

	print(cICA)

	if CLASS_BALANCE:
		ICA_T = (cICA == 1) ## True at indices where ICA = 1
		ICA_F = (cICA == 0)
		# print(ICA_T[24])
		# print(ICA_T[20])
		ICA_1 = cICA[ICA_T]
		ICA_0 = cICA[ICA_F]
		f_all_1 = []
		f_all_0 = []
		age_1 = age_all[ICA_T]
		age_0 = age_all[ICA_F]
		counter = 0
		for trfa in ICA_T:
			if trfa:
				f_all_1.append(f_all[counter])
			else:
				f_all_0.append(f_all[counter])
			counter += 1 
		# f_all_1 = f_all[ICA_T, :, :]
		# f_all_0 = f_all[ICA_F, :, :]
		trues = len(ICA_1)

		ICA_0 = ICA_0[0:trues]
		f_all_0 = f_all_0[0:trues]
		age_0 = age_0[0:trues]

		cICA = np.concatenate((ICA_0, ICA_1))
		f_all = np.concatenate((f_all_0, f_all_1))
		age_all = np.concatenate((age_0, age_1))
		print(len(f_all))

		f_all, f_dev, cICA, cICA_dev = train_test_split(f_all, cICA, random_state = seed, shuffle = True, test_size = DEV_SPLIT)

		kf = KFold(n_splits = SPLITS, shuffle = True)

		if not KFOLD:
			x_train, x_dev, age_train, age_dev, y_train, y_dev = train_test_split(f_all, age_all, cICA, random_state = seed, shuffle = True, test_size = TEST_FRACTION)
			x_train = np.array(x_train, dtype = float)
			x_dev = np.array(x_dev, dtype = float)
			age_train = np.array(age_train, dtype = float)
			age_dev = np.array(age_dev, dtype = float)
			y_train = np.array(y_train, dtype = float)
			y_dev = np.array(y_dev, dtype = float)
			print(np.shape(x_train))
			print(np.shape(x_dev))
			print(np.shape(y_train))
			print(np.shape(y_dev))
	else:
		x_train, x_dev, y_train, y_dev = train_test_split(f_all, cICA, random_state = seed, shuffle = True, test_size = TEST_FRACTION)

		# y_train = yf(y_train)
		# y_dev = yf(y_dev)

		x_train = np.array(x_train, dtype = float)
		x_dev = np.array(x_dev, dtype = float)
		y_train = np.array(y_train, dtype = float)
		y_dev = np.array(y_dev, dtype = float)
		print(np.shape(x_train))
		print(np.shape(x_dev))
		print(np.shape(y_train))
		print(np.shape(y_dev))


	if KFOLD:
		avg_train_accuracy = 0.0
		avg_test_accuracy = 0.0
		avg_train_f1_score = 0.0
		avg_test_f1_score = 0.0
		avg_actual_heavy = 0.0
		avg_precision = 0.0 
		k_counter = 1
		for train_index, test_index in kf.split(f_all):
			print("KFold step #" + str(k_counter))
			x_train, x_dev = f_all[train_index], f_all[test_index]
			y_train, y_dev = cICA[train_index], cICA[test_index]
			x_train = np.array(x_train, dtype = float)
			x_dev = np.array(x_dev, dtype = float)
			y_train = np.array(y_train, dtype = float)
			y_dev = np.array(y_dev, dtype = float)

			if TWO_D_CONV:
				x_train = np.expand_dims(x_train, axis = 3)
				x_dev = np.expand_dims(x_dev, axis = 3)
				y_train = np.expand_dims(y_train, axis = 3)
				y_dev = np.expand_dims(y_dev, axis = 3)
				print("2D Expansion")
				print(np.shape(x_train))
				print(np.shape(x_dev))
				print(np.shape(y_train))
				print(np.shape(y_dev))

			model = Sequential()
			#Watch for regularization
			if TWO_D_CONV:
				cnn_layer = keras.layers.Conv2D(NUM_OUTPUT_FILTERS, TWO_D_KERNEL_SIZE, strides = TWO_D_STRIDE, data_format = DATA_FORMAT, activation = CNN_ACTIVATION, use_bias = True, 
					input_shape = x_train[0].shape)
			else:
				cnn_layer = keras.layers.Conv1D(NUM_OUTPUT_FILTERS, KERNEL_SIZE, strides = STRIDE, data_format = DATA_FORMAT, activation = CNN_ACTIVATION, use_bias = True, input_shape = x_train[0].shape)
			
			lstm_layer = keras.layers.LSTM(OUTPUT_DIM, activation = RNN_ACTIVATION, dropout = DROPOUT, recurrent_dropout = RECURRENT_DROPOUT)

			model.add(cnn_layer)
			# model.add()
			model.add(lstm_layer)
			
			# model.add(TimeDistributed(Flatten()))

			# model.add(keras.layers.Dense(3, activation = 'hard_sigmoid'))
			# model.add(keras.layers.Dense(3, activation = 'tanh'))
			model.add(keras.layers.Dense(1, activation = FINAL_ACTIVATION))

			model.compile(optimizer = OPTIMIZER, loss = LOSS, metrics = METRICS)

			def sign(a):
				if a >= 0.5:
					return 1.0
				else:
					return 0.0

			sfunc = np.vectorize(sign)

			train_accuracy = []
			val_accuracy = []
			epoch = []
			for i in range(EPOCHS):
				print("Epoch " + str(i + 1) +  ":")
				model.fit(x_train, y_train, epochs = 1, batch_size = BATCH_SIZE)

				acc_train = model.evaluate(x_train, y_train)[1]
				acc_dev = model.evaluate(f_dev, cICA_dev)[1]
				train_accuracy.append(acc_train)
				val_accuracy.append(acc_dev)

				if abs(acc_train - acc_dev) <= (acc_dev + acc_train) / 20. and acc_train > ACC_THRESHOLD and acc_dev > ACC_THRESHOLD:
					break
				if abs(acc_train - acc_dev) >= (acc_dev + acc_train) / 20. and acc_train > MAX_ACC and acc_train > acc_dev:
					break
				if acc_train > MAX_ACC and acc_dev < MIN_ACC:
					break

				# print(model.evaluate(x_dev, y_dev)[1])
				# train_accuracy.append(model.evaluate(x_train, y_train)[1])
				# test_accuracy.append(model.evaluate(x_dev, y_dev)[1])
				# epoch.append(i)

				# predictions = model.predict(x_dev)
				# predictions = sfunc(predictions)
				# print(str(np.sum(y_dev[(predictions.squeeze() == 1)]) / float(np.sum(y_dev))) + " heavy drinkers were correctly predicted")
				# print(str(np.sum(y_dev[(predictions.squeeze() == 1)])/np.sum(predictions)) + " of predicted were correct")
				# recall = np.sum(y_dev[(predictions.squeeze() == 1)])/(np.sum(y_dev[(predictions.squeeze() == 1)]) + np.sum(y_dev[(predictions.squeeze() == 0)]))
				# precision = np.sum(y_dev[(predictions.squeeze() == 1)])/np.sum(predictions)
				# f1_score = 2*precision*recall/(precision+recall)

				# print("F1 Score is " + str(f1_score))
			# model.fit(x_train, y_train, epochs = EPOCHS, batch_size = BATCH_SIZE
				#validation_data = (x_dev, y_dev)
				# )
			# np.savetxt('./epoch' + str(k_counter) + '.txt', epoch)
			# np.savetxt('./train_acc_ydrop' + str(k_counter) + '.txt', train_accuracy)
			# np.savetxt('./val_acc_ydrop' + str(k_counter) + '.txt', val_accuracy)
			
			print("acc_train = " + str(acc_train) + "\nacc_dev = " + str(acc_dev))

			train_predictions = model.predict(x_train)

			predictions = model.predict(x_dev)

			print("raw")
			print(np.squeeze(predictions))
			print("converted")
			
			predictions = sfunc(predictions)
			train_predictions = sfunc(train_predictions)
			print(np.squeeze(predictions))
			print(np.squeeze(y_dev))
			print(len(predictions.squeeze() == y_dev))
			print(len(y_dev))
			train_counter = 0
			counter = 0
			for i in range(len(y_dev)):
				if predictions[i] == y_dev[i]:
					counter += 1
			for i in range(len(y_train)):
				if train_predictions[i] == y_train[i]:
					train_counter += 1
			print(str(counter / float(len(y_dev))) + " is the accuracy")
			print(str(np.sum(predictions)) + " is the count of predicted heavy drinkers")
			print("Of the " + str(np.sum(y_dev)) + " heavy drinkers, " + str(np.sum(y_dev[(predictions.squeeze() == 1)])) + " was classified as such")
			print(str(np.sum(y_dev[(predictions.squeeze() == 1)]) / float(np.sum(y_dev))) + " were correctly predicted")
			print(str(np.sum(y_dev[(predictions.squeeze() == 1)])/ np.sum(predictions)) + " of predicted were correct")
			print(str(np.sum(y_dev) / float(len(y_dev))) + " are actually heavy drinkers")

			train_score = model.evaluate(x_train, y_train)
			score = model.evaluate(x_dev, y_dev)
			print("The score is " + str(score))

			recall = np.sum(y_dev[(predictions.squeeze() == 1)])/(np.sum(y_dev[(predictions.squeeze() == 1)]) + np.sum(y_dev[(predictions.squeeze() == 0)]))
			precision = np.sum(y_dev[(predictions.squeeze() == 1)])/np.sum(predictions)
			f1_score = 2*precision*recall/(precision+recall)

			train_recall = np.sum(y_train[(train_predictions.squeeze() == 1)])/(np.sum(y_train[(train_predictions.squeeze() == 1)]) + np.sum(y_train[(train_predictions.squeeze() == 0)]))
			train_precision = np.sum(y_train[(train_predictions.squeeze() == 1)])/np.sum(train_predictions)
			train_f1_score = 2*train_precision*train_recall/(train_precision+train_recall)

			print("F1 Score is " + str(f1_score))
			print("Train F1 Score is " + str(train_f1_score))

			avg_train_accuracy += train_score[1]
			avg_test_accuracy += score[1]
			avg_train_f1_score += train_f1_score
			avg_test_f1_score += f1_score
			avg_actual_heavy += np.sum(y_dev) / float(len(y_dev))
			avg_precision += precision
			k_counter += 1

		avg_train_accuracy = avg_train_accuracy / SPLITS
		avg_test_accuracy = avg_test_accuracy / SPLITS
		avg_train_f1_score = avg_train_f1_score / SPLITS
		avg_test_f1_score = avg_test_f1_score / SPLITS
		avg_actual_heavy = avg_actual_heavy / SPLITS
		avg_precision = avg_precision / SPLITS

		print("KFOLD RESULTS: \n\n" + "Train accuracy: " + str(avg_train_accuracy) + "\nTrain f1_score: " + str(avg_train_f1_score))
		print("\nTest accuracy: " + str(avg_test_accuracy) + "\nTest f1_score: " + str(avg_test_f1_score))
		print("\nTest actual_heavy: " + str(avg_actual_heavy) + "\nTest precision: " + str(avg_precision))
	else:
		if CLASS_BALANCE_ADDITION:
			ICA_T = (y_train == 1) ## True at indices where ICA = 1
			ICA_F = (y_train == 0)
			ICA_1 = y_train[ICA_T]
			ICA_0 = y_train[ICA_F]
			ICA_1_data = x_train[ICA_T]
			ICA_0_data = x_train[ICA_F]
			trues = len(ICA_1)
			falses = len(ICA_0)
			ratio = falses // trues
			x_train_new = x_train[:]
			y_train_new = y_train[:]
			amount = ratio - 1
			for i in range(1):
				x_train_new = np.concatenate((x_train_new, ICA_1_data[:]))
				y_train_new = np.concatenate((y_train_new, ICA_1[:]))

			x_train, x_dummy, y_train, y_dummy = train_test_split(x_train_new, y_train_new, random_state = seed, shuffle = True, test_size = 0.0)

			x_train = np.array(x_train, dtype = float)
			y_train = np.array(y_train, dtype = float)

		if TWO_D_CONV:
			x_train = np.expand_dims(x_train, axis = 3)
			x_dev = np.expand_dims(x_dev, axis = 3)
			y_train = np.expand_dims(y_train, axis = 3)
			y_dev = np.expand_dims(y_dev, axis = 3)
			print("2D Expansion")
			print(np.shape(x_train))
			print(np.shape(x_dev))
			print(np.shape(y_train))
			print(np.shape(y_dev))

		model = Sequential()
		#Watch for regularization
		if TWO_D_CONV:
			cnn_layer = keras.layers.Conv2D(NUM_OUTPUT_FILTERS, TWO_D_KERNEL_SIZE, strides = TWO_D_STRIDE, data_format = DATA_FORMAT, activation = CNN_ACTIVATION, use_bias = True, input_shape = x_train[0].shape)
		else:
			cnn_layer = keras.layers.Conv1D(NUM_OUTPUT_FILTERS, KERNEL_SIZE, strides = STRIDE, data_format = DATA_FORMAT, activation = CNN_ACTIVATION, use_bias = True, input_shape = x_train[0].shape)
		
		# if INCLUDE_AGE:
		# 	print(age_train.shape)
		# 	age_layer = keras.layers.Dense(1, activation = AGE_ACTIVATION, input_shape = age_train.shape)
		# 	cnn_layer = Concatenate([cnn_layer, age_layer])
		# 	x_train = [x_train, age_train]
		# 	x_dev = [x_dev, age_dev]
		# 	print(x_dev)
		
		lstm_layer = keras.layers.LSTM(OUTPUT_DIM, activation = RNN_ACTIVATION)

		model.add(cnn_layer)
		# model.add()
		# model.add(lstm_layer)
		
		model.add(Flatten())

		model.add(keras.layers.Dense(1, activation = FINAL_ACTIVATION))

		if ADDITIONAL:
			for i in range(ADDITIONAL_LSTM):
				model.add(keras.layers.GRU(1, activation = RNN_ACTIVATION))

			model.add(keras.layers.Dense(1, activation = FINAL_ACTIVATION))
		model.compile(optimizer = OPTIMIZER, loss = LOSS, metrics = METRICS)

		def sign(a):
			if a >= 0.5:
				return 1.0
			else:
				return 0.0

		sfunc = np.vectorize(sign)

		for i in range(EPOCHS):
			print("Epoch " + str(i + 1) +  ":")
			model.fit(x_train, y_train, epochs = 1, batch_size = BATCH_SIZE)
			print(model.evaluate(x_dev, y_dev)[1])
			predictions = model.predict(x_dev)
			predictions = sfunc(predictions)
			print(str(np.sum(y_dev[(predictions.squeeze() == 1)]) / float(np.sum(y_dev))) + " heavy drinkers were correctly predicted")
			print(str(np.sum(y_dev[(predictions.squeeze() == 1)])/np.sum(predictions)) + " of predicted were correct")
			recall = np.sum(y_dev[(predictions.squeeze() == 1)])/(np.sum(y_dev[(predictions.squeeze() == 1)]) + np.sum(y_dev[(predictions.squeeze() == 0)]))
			precision = np.sum(y_dev[(predictions.squeeze() == 1)])/np.sum(predictions)
			f1_score = 2*precision*recall/(precision+recall)

			print("F1 Score is " + str(f1_score))
		# model.fit(x_train, y_train, epochs = EPOCHS, batch_size = BATCH_SIZE
			#validation_data = (x_dev, y_dev)
			# )
		
		predictions = model.predict(x_dev)

		print("raw")
		print(np.squeeze(predictions))
		print("converted")
		
		predictions = sfunc(predictions)
		print(np.squeeze(predictions))
		print(np.squeeze(y_dev))
		print(len(predictions.squeeze() == y_dev))
		print(len(y_dev))
		counter = 0
		for i in range(len(y_dev)):
			if predictions[i] == y_dev[i]:
				counter += 1
		print(str(counter / float(len(y_dev))) + " is the accuracy")
		print(str(np.sum(predictions)) + " is the count of predicted heavy drinkers")
		print("Of the " + str(np.sum(y_dev)) + " heavy drinkers, " + str(np.sum(y_dev[(predictions.squeeze() == 1)])) + " was classified as such")
		print(str(np.sum(y_dev[(predictions.squeeze() == 1)]) / float(np.sum(y_dev))) + " were correctly predicted")
		print(str(np.sum(y_dev[(predictions.squeeze() == 1)])/ np.sum(predictions)) + " of predicted were correct")
		print(str(np.sum(y_dev) / float(len(y_dev))) + " are actually heavy drinkers")

		score = model.evaluate(x_dev, y_dev)
		print("The score is " + str(score))

		recall = np.sum(y_dev[(predictions.squeeze() == 1)])/(np.sum(y_dev[(predictions.squeeze() == 1)]) + np.sum(y_dev[(predictions.squeeze() == 0)]))
		precision = np.sum(y_dev[(predictions.squeeze() == 1)])/np.sum(predictions)
		f1_score = 2*precision*recall/(precision+recall)

		print("F1 Score is " + str(f1_score))

		return counter / float(len(y_dev)), np.sum(y_dev[(predictions.squeeze() == 1)]) / float(np.sum(y_dev)), np.sum(y_dev[(predictions.squeeze() == 1)])/np.sum(predictions), np.sum(y_dev) / float(len(y_dev)), model
def main():
	avg_acc = 0.0
	avg_corr_pred = 0.0
	avg_corr_in_pred = 0.0	
	avg_actual = 0.0
	if not KFOLD:
		for i in range(RANDOMS):
			np.random.seed(i)
			acc, corr_pred, corr_in_pred, actual, model = main_sub(i)
			avg_acc += acc / RANDOMS
			avg_corr_pred += corr_pred / RANDOMS
			avg_corr_in_pred += corr_in_pred / RANDOMS
			avg_actual += actual / RANDOMS

		

		print(str(avg_acc) + " is the AVG accuracy")
		print(str(avg_corr_pred) + " were AVG correctly predicted")
		print(str(avg_corr_in_pred) + " of AVG predicted were correct")
		print(str(avg_actual) + " AVG are actually heavy drinkers")
	else:
		main_sub(0)

main()