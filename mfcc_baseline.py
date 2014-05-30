import numpy as np 
import pandas as pd 

from features import mfcc
from features import logfbank

import scipy.io.wavfile as wav

import csv


def read_data(files_amount, total_length, nc = 13, path=''):
	for i in range(files_amount):
		(rate,sig) = wav.read(path + str(i) + '.wav')
		mfcc_feat = mfcc(sig,rate, numcep = nc)
		mfcc_feat = np.reshape(mfcc_feat, (len(mfcc_feat)*nc, 1))

		if any(np.isnan(mfcc_feat)) or any(np.isinf(mfcc_feat)):
			ind = [x for x in range(len(mfcc_feat)) if np.isnan(mfcc_feat[x]) or np.isinf(mfcc_feat[x])]
			for x in ind:
				mfcc_feat[x] = 0

		if i == 0:
			mfcc_data = mfcc_feat

		if i != 0:
			if len(mfcc_feat) == total_length:
				mfcc_data = np.hstack((mfcc_data, mfcc_feat))
			else:
				if len(mfcc_feat) > total_length:
					mfcc_data = np.hstack((mfcc_data, mfcc_feat[:(total_length)]))
				else:
					xx = np.vstack((mfcc_feat,np.reshape(np.asarray([0] * (total_length-len(mfcc_feat))), (total_length-len(mfcc_feat),1) )))
					mfcc_data = np.hstack((mfcc_data, xx))

	return mfcc_data


def write_submission(pred_labels, file_name = 'mfcc_baseline.csv'):
	with open(file_name, 'w') as fp:
		a = csv.writer(fp, delimiter = ',')
		a.writerow(['Id', 'Category'])
		for i in range(len(pred_labels)):
			a.writerow([i, pred_labels[i]])


data = np.asarray(pd.read_csv('train_answers.csv'))
train_y = data[:,1]

mfcc_train = []
mfcc_test = []

nc = 13 # num cepstrum
proper_length = 2994
total_length = nc * proper_length

mfcc_train = read_data(200, total_length, nc)
mfcc_test = read_data(800, total_length, nc, 'test/')


from sklearn.svm import LinearSVC 

mfcc_train = np.swapaxes(mfcc_train, 0, 1)
mfcc_test = np.swapaxes(mfcc_test, 0, 1)

model = LinearSVC()

enumerated = dict(enumerate(train_y))
ty = {}

for a in enumerated.keys():
	ty[enumerated[a]] = a

y = [ty[x] for x in train_y]

model.fit(mfcc_train, np.asarray([int(x) for x in y]))

pred = model.predict(mfcc_test)

pred_labels = [enumerated[x] for x in pred]

write_submission(pred_labels)



