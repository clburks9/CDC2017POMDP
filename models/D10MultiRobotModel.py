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
from copy import deepcopy
'''
****************************************************
File: D10MulitRobotModel.py
Written By: Luke Burks
October 2018

Container Class for problem specific models
Model: 5 2D Robots, each trying to reach a specific 
location
The "Multi-Robot 2D Hallway Problem"

Bounded 0,10 for each dimension

Order of dimensions, x,y for each robot
Robot 0 is special 

****************************************************
'''


class ModelSpec:

	def __init__(self):
		self.fileNamePrefix = 'D10MultiRobot'; 
		self.STM = None
		self.acts = 21; 
		self.obs = 84; 


	def buildTransition(self):
		self.bounds = []; 
		for i in range(0,10):
			self.bounds.append([0,10]); 

		#Order of actions:
		#left,right,up,down for each robot, final action is special
		#for every action, need a 10 element 1 hot vector
		delta = 1;
		self.delA = [] 
		for rob in range(0,5):
			#left
			tmp = [0]*10; 
			tmp[rob*2 + 0] = -delta; 
			self.delA.append(tmp); 

			#right
			tmp = [0]*10; 
			tmp[rob*2 + 0] = delta; 
			self.delA.append(tmp); 

			#up
			tmp = [0]*10; 
			tmp[rob*2 + 1] = delta; 
			self.delA.append(tmp); 

			#down
			tmp = [0]*10; 
			tmp[rob*2 + 1] = -delta; 
			self.delA.append(tmp); 

		#ping
		tmp = [0]*10
		self.delA.append(tmp); 
		

		#Need a separate 10x10 matrix for each action...
		#There's no reason all robots twitch when one moves

		#go with .5 noise on relevant dimensions
		self.delAVar = np.zeros(shape=(21,10,10)).tolist(); 
		for a in range(0,21):
			for i in range(0,10):
				self.delAVar[a][i][i] = 0.001; 
			if(a<20):
				self.delAVar[a][2*(a//4)][2*(a//4)] = .5; 
				self.delAVar[a][2*(a//4)+1][2*(a//4)+1] = .5; 

		self.discount = 0.95; 

	#Problem Specific
	def buildObs(self,gen=True):
		

		if(gen):
			
			#for each 
			weight = [[0,1],[-1,1],[1,1],[0,2],[0,0]]
			bias = [1,0,0,0,0]; 
			steep = 2;
			weight = (np.array(weight)*steep).tolist(); 
			bias = (np.array(bias)*steep).tolist(); 
			self.pz = Softmax(weight,bias); 
			# print('Plotting Observation Model'); 
			# [x,y,dom] = self.pz.plot2D(low=[-10,-10],high=[10,10],vis=False); 
			# plt.contourf(x,y,dom); 
			# plt.colorbar(); 
			# plt.show(); 
						

			np.save("../models/obs/"+ self.fileNamePrefix + "OBS.npy",self.pz);
		else:
			self.pz = np.load("../models/obs/"+ self.fileNamePrefix + "OBS.npy").tolist(); 


	
	def buildReward(self,gen = True):
		if(gen): 

			self.r = [0]*len(self.delA);
			
			#square of desired poses with robot 0 in the middle
			#might need to mess with this for observability

			#Just make every action give the same reward
			goal = [5,5,2.5,7.5,7.5,7.5,7.5,2.5,2.5,2.5]; 

			for i in range(0,len(self.r)):
				self.r[i] = GM();  
				for j in range(1,7):
					self.r[i].addG(Gaussian(goal,(j*2*np.identity(10)).tolist(),1))

			self.r[0].display(); 
			
			np.save("../models/rew/"+ self.fileNamePrefix + "REW.npy",self.r);

		else:
			self.r = np.load("../models/rew/"+ self.fileNamePrefix + "REW.npy").tolist();




def TestSoftmaxIn10D():
	#Octohedron Specs
	numClasses = 9; 
	boundries = []; 
	for i in range(1,numClasses):
		boundries.append([i,0]); 
	B = np.matrix([-1,-1,0.5,-1,-1,1,0.5,-1,1,1,0.5,-1,1,-1,0.5,-1,-1,-1,-0.5,-1,-1,1,-0.5,-1,1,1,-0.5,-1,1,-1,-0.5,-1]).T; 
	#B = np.matrix([-1,-1,2,-1,-1,1,2,-1,1,1,2,-1,1,-1,2,-1,-1,-1,-2,-1,-1,1,-2,-1,1,1,-2,-1,1,-1,-2,-1]).T; 
	
	#Cube Specs
	numClasses = 7; 
	boundries = [[1,0],[2,0],[3,0],[4,0],[5,0],[6,0]]; 
	# B = np.matrix(np.concatenate(([-1,-1,2])))
	# B = np.matrix(np.concatenate(([-1,-1,2,-1],[-1,1,2,-1],[1,1,2,-1],[1,-1,2,-1],[-1,-1,-2,-1],[-1,1,-2,-1],[1,1,-2,-1],[1,-1,-2,-1]))).T; 
	B = np.matrix(np.concatenate(([0,-1,0,-1],[0,0,1,-1],[1,0,0,-1],[0,0,1,-1],[-1,0,0,-1],[0,1,0,-1]))).T;


	#Tesseract, in 4D
	numClasses = 9; 
	boundries = []; 
	for i in range(1,numClasses):
		boundries.append([i,0]); 
	#B = np.matrix(np.concatenate(([1,0,0,0,-1],[-1,0,0,0,-1],[0,1,0,0,-1],[0,-1,0,0,-1],[0,0,1,0,-1],[0,0,-1,0,-1],[0,0,0,1,-1],[0,0,0,-1,-1],))).T;
	#pose = [1,-1,-1,0]; 
	B = np.matrix(np.concatenate(([1,0,0,0,-2],[-1,0,0,0,0],[0,1,0,0,0],[0,-1,0,0,-2],[0,0,1,0,0],[0,0,-1,0,-2],[0,0,0,1,-1],[0,0,0,-1,-1]))).T;


	#Parrallel-Piped, 4D, Tilted along X and Y
	#In Progress
	numClasses = 9; 
	boundries = []; 
	for i in range(1,numClasses):
		boundries.append([i,0]); 
	B = np.matrix(np.concatenate(([-1,1,0,0,-1],[1,-1,0,0,-1],[-1,1,0,0,-1],[1,-1,0,0,-1],[0,0,1,0,-1],[0,0,-1,0,-1],[0,0,0,1,-1],[0,0,0,-1,-1]))).T;



	#Notes:
	#There's a push and pull between constants on the same dimension
	#Constants need to both be negative on the same dimension
	#The absolute value of the sum of the constants needs to be <= the sum of the absolute values of the normals
	#higher values squeeze against their direction if constants are negative

	pz = Softmax(); 
	pz.buildGeneralModel(dims=4,numClasses=numClasses,boundries=boundries,B=B,steepness=10); 

	pz2 = Softmax(deepcopy(pz.weights),deepcopy(pz.bias));
	pz3 = Softmax(deepcopy(pz.weights),deepcopy(pz.bias));
	pz4 = Softmax(deepcopy(pz.weights),deepcopy(pz.bias));

	for i in range(0,len(pz2.weights)):
		pz2.weights[i] = [pz2.weights[i][0],pz2.weights[i][2]]

	for i in range(0,len(pz3.weights)):
		pz3.weights[i] = [pz3.weights[i][1],pz3.weights[i][2]]

	for i in range(0,len(pz4.weights)):
		pz4.weights[i] = [pz4.weights[i][0],pz4.weights[i][1]]

	fig = plt.figure(); 
	[x,y,c] = pz2.plot2D(low=[-5,-5],high=[5,5],vis = False); 
	plt.contourf(x,y,c); 
	plt.xlabel('X Axis'); 
	plt.ylabel('Z Axis'); 
	plt.title('Slice Across Y and W Axis')
	plt.axis('equal')

	fig = plt.figure(); 
	[x,y,c] = pz3.plot2D(low=[-5,-5],high=[5,5],vis = False); 
	plt.contourf(x,y,c); 
	plt.xlabel('Y Axis'); 
	plt.ylabel('Z Axis');
	plt.title('Slice Across X and W axis')
	plt.axis('equal')

	fig = plt.figure(); 
	[x,y,c] = pz4.plot2D(low=[-5,-5],high=[5,5],vis = False); 
	plt.contourf(x,y,c); 
	plt.xlabel('X Axis'); 
	plt.ylabel('Y Axis');
	plt.title('Slice Across Z and W Axis'); 
	plt.axis('equal')

	plt.show();

	#pz.plot3D();  


if __name__ == '__main__':
	a = ModelSpec(); 
	# a.buildTransition(); 
	# a.buildReward(gen = False); 
	# a.buildObs(gen = False); 

	TestSoftmaxIn10D(); 







