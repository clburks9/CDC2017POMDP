from __future__ import division
import numpy as np
from scipy.stats import multivariate_normal as mvn
import random
import copy
import cProfile
import re
import matplotlib.pyplot as plt
import math
from scipy.stats import norm
import os; 
from math import sqrt
import signal
import sys
import cProfile
sys.path.append('../src/'); 
from gaussianMixtures import Gaussian
from gaussianMixtures import GM
import matplotlib.animation as animation
from numpy import arange
import time
import matplotlib.image as mgimg
from softmaxModels import Softmax



'''
****************************************************
File: D2ColinearSoftmaxModel.py
Written By: Luke Burks
October 2018

Container Class for problem specific models
Model: Cop and Robber, both in 1D
The "Colinear Robots" problem

Reworked for TRO-Paper

Bounds from 0 to 5 on both dimensions 
****************************************************


'''




class ModelSpec:

	def __init__(self):
		self.fileNamePrefix = 'D2ColinearSoftmax'; 
		self.acts = 3; 
		self.obs = 2;
		self.STM = None 


	#Problem specific
	def buildTransition(self):
		self.bounds = [[0,5],[0,5]]; 
		self.delAVar = [[0.01,0],[0,.5]]; 
		delta = 0.5; 
		self.delA = [[-delta,0],[delta,0],[0,0]]; 
		self.discount = 0.95; 

	#Problem Specific
	def buildObs(self,gen=True):
		#detect, no detect

		if(gen):
			weight = [[-1.3926,1.3926],[-0.6963,0.6963],[0,0]];
			bias = [0,.4741,0]; 
			self.pz = Softmax(weight,bias);


			print('Plotting Observation Model'); 
			[x,y,dom] = self.pz.plot2D(low=[0,0],high=[5,5],vis=False); 
			plt.contourf(x,y,dom); 
			plt.colorbar(); 
			plt.show(); 



			np.save(os.path.dirname(__file__) + '/' + "../models/obs/"+ self.fileNamePrefix + "OBS.npy",self.pz);
		else:
			self.pz = np.load(os.path.dirname(__file__) + '/' + "../models/obs/"+ self.fileNamePrefix + "OBS.npy").tolist(); 


	#Problem Specific
	def buildReward(self,gen = True):
		if(gen): 

			self.r = [0]*len(self.delA);
			

			for i in range(0,len(self.r)):
				self.r[i] = GM();  

			#var = (np.identity(2)*.5).tolist(); 
			var = [[1,.8],[.8,1]]; 

			for i in range(0,len(self.r)):
				for j in range(-1,7):
					self.r[i].addG(Gaussian([j-self.delA[i][0],j-self.delA[i][1]],var,3));

			

			print('Plotting Reward Model'); 
			for i in range(0,len(self.r)):
				self.r[i].plot2D(high = [5,5],low = [0,0],xlabel = 'Cop',ylabel = 'Robber',title = 'Reward for action: ' + str(i)); 


			np.save(os.path.dirname(__file__) + '/' + "../models/rew/"+ self.fileNamePrefix + "REW.npy",self.r);

		else:
			self.r = np.load(os.path.dirname(__file__) + '/' + "../models/rew/"+ self.fileNamePrefix + "REW.npy").tolist();


if __name__ == '__main__':
	a = ModelSpec(); 
	a.buildTransition(); 
	a.buildReward(gen = False); 
	a.buildObs(gen = False); 
	