from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from gaussianMixtures import GM,Gaussian






if __name__ == '__main__':
	
	data = {};

	data['GM'] = np.load('../results/D2Diffs/D2Diffs_Data1.npy',encoding='latin1').tolist(); 
	data['VB'] = np.load('../results/D2DiffsSoftmax/D2DiffsSoftmax_Data1.npy',encoding='latin1').tolist(); 
	data['Greedy'] = np.load('../results/D2DiffsSoftmax/D2DiffsSoftmax_Data_Greedy1.npy',encoding='latin1').tolist(); 

	keys = ['GM','VB','Greedy']; 

	for key in keys:

		rew = data[key]['Rewards']; 
		suma = 0; 
		for r in rew:
			suma += r[-1]; 
		suma /= len(rew); 
		print(key,suma); 