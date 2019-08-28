

import os

import csv

import numpy as np


if not os.path.exists('draw_csv'):
	os.makedirs('draw_csv')


def create_csv(filepath):
	file=open(filepath).read().splitlines()
	data=dict()
	current_dataset_name = ""
	current_dataset_array = list()
	data_test = dict()
	for i in range(len(file)):
		string = file[i].split()
		if "experiment:" in string:
			# firstline : !!! experiment: xth ....
			# second line: Experiment of XXX ...
			name = file[i+1].split()[2]
			if name not in data:
				# initialize
				data[name]=list()
				data_test[name]=list()
			if len(current_dataset_name)==0:
				current_dataset_name=name
			else:
				# encounter new experiment, append recorded values
				data[current_dataset_name].append(current_dataset_array)
				current_dataset_array=list()
				current_dataset_name=name
		
		if "loss:" in string:
			if "val_loss:" not in string:  # exclude the evaluate step
				pass
			else:
				# recording
				numbers = list()
				for s in string:
					try:
						x=float(s)
						numbers.append(x)
					except:
						pass
				current_dataset_array.append(numbers)
		if len(string)==2 and string[0][0]=="[" and string[-1][-1] =="]":
			# this is the case when it reads the score for current experiment
			temp=[]
			temp.append(float(string[0][1:-1])) # evaluate loss value
			temp.append(float(string[1][:-1])) # evaluate accuracy value
			data_test[current_dataset_name].append(temp)

	# append last recorded values
	data[current_dataset_name].append(current_dataset_array)

	# take average
	data_ave = dict()
	for key in data:
		data_ave[key] = np.mean(data[key], axis = 0)



	for key in data:
		with open('draw_csv/'+key+".csv","w") as csvfile:
			filewriter = csv.writer(csvfile, delimiter=',',
							quotechar='|', quoting=csv.QUOTE_MINIMAL)
			filewriter.writerow(["epoch","loss","acc","val_loss","val_acc"])
			for i in range(data_ave[key].shape[0]):
				filewriter.writerow([i+1]+data_ave[key][i].tolist())



	with open("draw_csv/test_result.csv","w") as csvfile:
		filewriter = csv.writer(csvfile, delimiter=',',
						quotechar='|', quoting=csv.QUOTE_MINIMAL)
		filewriter.writerow(['experiment_name','test_loss','test_acc'])
		for key in data:
			for i in range(len(data_test[key])):
				filewriter.writerow([key+' experiment '+str(i)+'th']+data_test[key][i])


if __name__ == '__main__':
	file = input('please input the filepath/name of the file: ')
	while True:
		try:
			_ = open(file)
			break
		except:
			print('error')
			file = input('please input the filepath/name of the file: ')

	create_csv(file)
























