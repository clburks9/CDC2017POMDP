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
File: D2Diffs_2SoftmaxModel.py
Written By: Luke Burks
October 2018

Container Class for problem specific models
Model: Cop and Robber, both in 2D, represented
by differences in x and y dims
The "Field of Tall Grass" problem, the cop knows 
where he is, but can't see the robber

Uses softmax observations

Bounds from 0 to 5 on both dimensions 
****************************************************


'''

__author__ = "Luke Burks"
__copyright__ = "Copyright 2018, Cohrint"
__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Luke Burks"
__email__ = "clburks9@gmail.com"
__status__ = "Development"



class ModelSpec:

	def __init__(self):
		self.fileNamePrefix = 'D2DiffsSoftmax'; 
		self.STM = None
		self.acts = 5; 
		self.obs = 5; 

	#Problem specific
	def buildTransition(self):
		self.bounds = [[-10,10],[-10,10]]; 
		self.delAVar = (np.identity(2)*1).tolist(); 
		self.delAVar[0][0] = .25; 
		#self.delA = [[-0.5,0],[0.5,0],[0,-0.5],[0,0.5],[0,0],[-0.5,-0.5],[0.5,-0.5],[-0.5,0.5],[0.5,0.5]]; 
		delta = 1; 
		self.delA = [[-delta,0],[delta,0],[0,delta],[0,-delta],[0,0]]; 
		self.discount = 0.95; 



	#Problem Specific
	def buildObs(self,gen=True):
		#cardinal + 1 model
		#left,right,up,down,near

		if(gen):
			
			weight = [[0,1],[-1,1],[1,1],[0,2],[0,0]]
			bias = [1,0,0,0,0]; 
			steep = 2;
			weight = (np.array(weight)*steep).tolist(); 
			bias = (np.array(bias)*steep).tolist(); 
			self.pz = Softmax(weight,bias); 
			print('Plotting Observation Model'); 
			[x,y,dom] = self.pz.plot2D(low=[-10,-10],high=[10,10],vis=False); 
			plt.contourf(x,y,dom); 
			plt.colorbar(); 
			plt.show(); 
						

			np.save("../models/obs/"+ self.fileNamePrefix + "OBS.npy",self.pz);
		else:
			self.pz = np.load("../models/obs/"+ self.fileNamePrefix + "OBS.npy").tolist(); 
			
			
	#Problem Specific
	def buildReward(self,gen = True):
		if(gen): 

			self.r = [0]*len(self.delA);
			

			for i in range(0,len(self.r)):
				self.r[i] = GM();  

			var = (np.identity(2)*.5).tolist(); 

			for i in range(0,len(self.r)):
				self.r[i].addG(Gaussian([-self.delA[i][0],-self.delA[i][1]],var,10));

			

			print('Plotting Reward Model'); 
			for i in range(0,len(self.r)):
				self.r[i].plot2D(high = [10,10],low = [-10,-10],xlabel = 'Robot X',ylabel = 'Robot Y',title = 'Reward for action: ' + str(i)); 

			print('Condensing Reward Model');
			for i in range(0,len(self.r)):
				self.r[i] = self.r[i].kmeansCondensationN(k = 5);


			print('Plotting Condensed Reward Model'); 
			for i in range(0,len(self.r)):
				#self.r[i].plot2D(xlabel = 'Robot X',ylabel = 'Robot Y',title = 'Reward for action: ' + str(i)); 
				[x,y,c] = self.r[i].plot2D(high = [10,10],low = [-10,-10],vis = False);  
	
				minim = np.amin(c); 
				maxim = np.amax(c); 

				#print(minim,maxim); 
				levels = np.linspace(minim,maxim); 
				plt.contourf(x,y,c,levels = levels,vmin = minim,vmax = maxim,cmap = 'viridis');
				plt.title('Reward for action: ' + str(i));
				plt.xlabel('Robot X'); 
				plt.ylabel('Robot Y'); 
				plt.show(); 


			np.save("../models/rew/"+ self.fileNamePrefix + "REW.npy",self.r);

		else:
			self.r = np.load("../models/rew/"+ self.fileNamePrefix + "REW.npy").tolist();




if __name__ == '__main__':
	a = ModelSpec(); 
	a.buildTransition(); 
	a.buildReward(gen = False); 
	a.buildObs(gen = False); 

	#print(a.pz.pointEvalND(0,[0,0])); 

	

	



