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


'''
****************************************************
File: D4DiffsModel.py
Written By: Luke Burks
January 2018

Container Class for problem specific models
Model: Cop and Robber, both in 2D, represented
by differences in x and y dims
The "Field of Tall Grass" problem, the cop knows 
where he is, but can't see the robber

4 dimensional state for NCV robber
S = {delX,delY,delXdot,delYdot}

Robber nominal velocity is 0.25

Bounds from 0 to 5 on both dimensions 
****************************************************


'''

__author__ = "Luke Burks"
__copyright__ = "Copyright 2017, Cohrint"
__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Luke Burks"
__email__ = "clburks9@gmail.com"
__status__ = "Development"



class ModelSpec:

	def __init__(self,mode = "NCP"):
		self.fileNamePrefix = 'D4Diffs'; 
		self.acts = 5; 
		self.obs = 5; 
		self.STM = None;
		self.mode=mode; 

	#Problem specific
	def buildTransition(self):
		self.bounds = [[-10,10],[-10,10],[-0.25,0.25],[-0.25,0.25]]; 
		self.delAVar = (np.identity(4)*1).tolist(); 

		#Linear Dynamics
		if(self.mode=='NCV'):
			self.STM = np.matrix([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]]); 
		else:
			self.STM = np.matrix([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]); 

		#Target Velocity doesn't change much
		self.delAVar[2][2] = 0.00001; 
		self.delAVar[3][3] = 0.00001; 
		
		#cops actions don't change robber velocity
		#Cop assumed to have 0 velocity
		delta = 1; 
		self.delA = [[-delta,0,0,0],[delta,0,0,0],[0,delta,0,0],[0,-delta,0,0],[0,0,0,0]]; 
		self.discount = 0.95; 
		

	#Problem Specific
	def buildObs(self,gen=True):
		#cardinal + 1 model
		#left,right,up,down,near

		if(gen):
			self.pz = [0]*5; 
			for i in range(0,5):
				self.pz[i] = GM(); 
			var = (np.identity(4)*1).tolist(); 
			
			var[2][2] = 10; 
			var[3][3] = 10; 
			
			#weight correction for additional dimensions
			weightCorrection = 1/0.0159143090766; 

			for x in range(-10,11):
				for y in range(-10,11):

					#left
					if(x<-1 and abs(x) > abs(y)):
						self.pz[0].addG(Gaussian([x,y,0,0],var,weightCorrection)); 
					#right
					if(x>1 and abs(x) > abs(y)):
						self.pz[1].addG(Gaussian([x,y,0,0],var,weightCorrection)); 
					#up
					if(y>1 and abs(x) < abs(y)):
						self.pz[2].addG(Gaussian([x,y,0,0],var,weightCorrection));
					#down
					if(y<-1 and abs(y) > abs(x)):
						self.pz[3].addG(Gaussian([x,y,0,0],var,weightCorrection)); 
					if(x>= -1 and x<=1 and y>=-1 and y<=1):
						self.pz[4].addG(Gaussian([x,y,0,0],var,weightCorrection)); 



			# c = np.zeros(shape=(21,21)).tolist(); 
			# for i in range(-10,11):
			# 	for j in range(-10,11):
			# 		for k in range(0,5):
			# 			c[i+10][j+10] += self.pz[k].pointEval([j,i,-0.25,-0.25]); 
			# plt.contourf(c,cmap='viridis'); 
			# plt.colorbar(); 
			# plt.show(); 

			# print('Plotting Observation Models'); 
			# for i in range(0,len(self.pz)):
			# 	#self.pz[i].plot2D(xlabel = 'Robot X',ylabel = 'Robot Y',title = 'Obs:' + str(i)); 
			# 	[x,y,c] = self.pz[i].plot2D(low = [-10,-10],high = [10,10],vis=False); 
			# 	plt.contourf(x,y,c,cmap='viridis'); 
			# 	plt.colorbar(); 
			# 	plt.show(); 

			print('Condensing Observation Models'); 
			for i in range(0,len(self.pz)):
				#self.pz[i] = self.pz[i].kmeansCondensationN(64);
				self.pz[i].condense(15);  

			# c = np.zeros(shape=(21,21)).tolist(); 
			# for i in range(-10,11):
			# 	for j in range(-10,11):
			# 		for k in range(0,5):
			# 			c[i+10][j+10] += self.pz[k].pointEval([j,i,-0.25,-0.25]); 
			# plt.contourf(c,cmap='viridis'); 
			# plt.colorbar(); 
			# plt.show(); 


			# print('Plotting Condensed Observation Models'); 
			# for i in range(0,len(self.pz)):
			# 	#self.pz[i].plot2D(xlabel = 'Robot X',ylabel = 'Robot Y',title = 'Obs:' + str(i)); 
			# 	[x,y,c] = self.pz[i].plot2D(low = [-10,-10],high = [10,10],vis=False); 
			# 	plt.contourf(x,y,c,cmap='viridis'); 
			# 	plt.colorbar(); 
			# 	plt.show(); 


			f = open(os.path.dirname(__file__) + '/' + "../models/obs/"+ self.fileNamePrefix + "OBS.npy","w"); 
			np.save(f,self.pz);
		else:
			self.pz = np.load(os.path.dirname(__file__) + '/' + "../models/obs/"+ self.fileNamePrefix + "OBS.npy").tolist(); 

			
	#Problem Specific
	def buildReward(self,gen = True):
		if(gen): 

			self.r = [0]*len(self.delA);
			

			for i in range(0,len(self.r)):
				self.r[i] = GM();  

			var = (np.identity(4)*.5).tolist(); 

			for i in range(0,len(self.r)):
				self.r[i].addG(Gaussian([-self.delA[i][0],-self.delA[i][1],0,0],var,100));


			# print('Plotting Reward Model'); 
			# for i in range(0,len(self.r)):
			# 	self.r[i].plot2D(high = [10,10],low = [-10,-10],xlabel = 'Robot X',ylabel = 'Robot Y',title = 'Reward for action: ' + str(i)); 

			# print('Condensing Reward Model');
			# for i in range(0,len(self.r)):
			# 	self.r[i] = self.r[i].kmeansCondensationN(k = 5);


			# print('Plotting Condensed Reward Model'); 
			# for i in range(0,len(self.r)):
			# 	#self.r[i].plot2D(xlabel = 'Robot X',ylabel = 'Robot Y',title = 'Reward for action: ' + str(i)); 
			# 	[x,y,c] = self.r[i].plot2D(high = [10,10],low = [-10,-10],vis = False);  
	
			# 	minim = np.amin(c); 
			# 	maxim = np.amax(c); 
			# 	print(maxim); 
			# 	#print(minim,maxim); 
			# 	levels = np.linspace(minim,maxim); 
			# 	plt.contourf(x,y,c,levels = levels,vmin = minim,vmax = maxim,cmap = 'viridis');
			# 	plt.title('Reward for action: ' + str(i));
			# 	plt.xlabel('Robot X'); 
			# 	plt.ylabel('Robot Y'); 
			# 	plt.show(); 


			f = open(os.path.dirname(__file__) + '/' + "../models/rew/"+ self.fileNamePrefix + "REW.npy","w"); 
			np.save(f,self.r);

		else:
			self.r = np.load(os.path.dirname(__file__) + '/' + "../models/rew/"+ self.fileNamePrefix + "REW.npy").tolist();


if __name__ == '__main__':
	a = ModelSpec(); 
	a.buildTransition(); 
	a.buildReward(gen = True); 
	a.buildObs(gen = True); 
	
	

	



