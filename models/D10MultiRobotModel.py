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
File: D6MulitRobotModel.py
Written By: Luke Burks
November 2018

Test for the 10D problem
Container Class for problem specific models
Model: 3 2D Robots, each trying to reach a specific 
location
The "Multi-Robot 2D Hallway Problem"

Bounded -5,5 for each dimension

Order of dimensions, x,y for each robot
Robot 0 is special 

****************************************************
'''


class ModelSpec:

	def __init__(self):
		self.fileNamePrefix = 'D10MultiRobot'; 
		self.STM = None
		#Acts: 4 moves per robot
		self.acts = 20; 
		self.obs = 5;
		self.obs2 = 5; 
		self.obs3 = 5;  
		self.obs4 = 5; 
		self.obs5 = 5; 


	def buildTransition(self):
		self.bounds = []; 
		for i in range(0,10):
			self.bounds.append([-5,5]); 

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


		self.seconddelA = []; 
		for rob in range(0,5):
			#left
			tmp = [0]*10; 
			tmp[rob*2 + 0] = -delta; 
			tmp[rob*2 + 1] = -delta/2; 
			self.seconddelA.append(tmp); 

			#right
			tmp = [0]*10; 
			tmp[rob*2 + 0] = delta; 
			tmp[rob*2 + 1] = -delta/2; 
			self.seconddelA.append(tmp); 

			#up
			tmp = [0]*10; 
			tmp[rob*2 + 1] = delta; 
			tmp[rob*2 + 0] = -delta/2; 
			self.seconddelA.append(tmp); 

			#down
			tmp = [0]*10; 
			tmp[rob*2 + 1] = -delta; 
			tmp[rob*2 + 0] = -delta/2; 
			self.seconddelA.append(tmp); 

		
		self.thirddelA = []; 
		for rob in range(0,5):
			#left
			tmp = [0]*10; 
			tmp[rob*2 + 0] = -delta; 
			tmp[rob*2 + 1] = delta/2; 
			self.thirddelA.append(tmp); 

			#right
			tmp = [0]*10; 
			tmp[rob*2 + 0] = delta; 
			tmp[rob*2 + 1] = delta/2; 
			self.thirddelA.append(tmp); 

			#up
			tmp = [0]*10; 
			tmp[rob*2 + 1] = delta; 
			tmp[rob*2 + 0] = delta/2; 
			self.thirddelA.append(tmp); 

			#down
			tmp = [0]*10; 
			tmp[rob*2 + 1] = -delta; 
			tmp[rob*2 + 0] = delta/2; 
			self.thirddelA.append(tmp);


		#print(self.delA); 

		#Need a separate 10x10 matrix for each action...
		#There's no reason all robots twitch when one moves

		#go with .5 noise on relevant dimensions
		self.delAVar = np.zeros(shape=(20,10,10)).tolist(); 
		for a in range(0,20):
			for i in range(0,10):
				self.delAVar[a][i][i] = 0.00001; 
			if(a<12):
				self.delAVar[a][2*(a//4)][2*(a//4)] = .01; 
				self.delAVar[a][2*(a//4)+1][2*(a//4)+1] = .01; 
		

		# print(self.delA); 
		# print(np.array(self.delAVar)); 

		self.discount = 0.95; 

	#Problem Specific
	def buildObs(self,gen=True):
		
		#if no ping action was taken, then no observation

		if(gen):
			
			#for each 
			#Parrallel-Piped, 4D, Tilted along X and Y
			numClasses = 9; 
			boundries = []; 
			for i in range(1,numClasses):
				boundries.append([i,0]); 
			B = np.matrix(np.concatenate(([-1,1,0,0,-1],[1,-1,0,0,-1],[-1,1,0,0,-1],[1,-1,0,0,-1],[0,0,-1,1,-1],[0,0,1,-1,-1],[0,0,-1,1,-1],[0,0,1,-1,-1]))).T;

			#Robot1, observes landmark
			cent = [0,0]; 
			length = 2; 
			width = 2; 
			orient = 0; 
			steep = .1; 

			self.pz1 = Softmax(); 
			self.pz1.buildOrientedRecModel(cent,orient,length,width,steepness=steep);
			
			#add zeros for 1,2,4,5,6,7,8,9
			for i in range(0,len(self.pz1.weights)):
				self.pz1.weights[i].insert(1,0);
				self.pz1.weights[i].insert(2,0);  
				self.pz1.weights[i].insert(3,0);
				self.pz1.weights[i].insert(4,0); 
				#self.pz1.weights[i].append(0); 
				self.pz1.weights[i].append(0);  
				self.pz1.weights[i].append(0);
				self.pz1.weights[i].append(0); 
				self.pz1.weights[i].append(0);  
			#print(self.pz1.weights[1]); 
			#self.pz1.bias.insert(1,0);
			#self.pz1.bias.insert(2,0);
			#self.pz1.bias.append(0); 
			#self.pz1.bias.append(0); 
			
			#Robot2, observes relative to robot 1
			self.pz2 = Softmax(); 
			self.pz2.buildGeneralModel(dims=4,numClasses=numClasses,boundries=boundries,B=B,steepness=steep); 

			#add zeros for 2,3,4,7,8,9
			for i in range(0,len(self.pz2.weights)):
				self.pz2.weights[i].insert(2,0);
				self.pz2.weights[i].insert(3,0);
				self.pz2.weights[i].insert(4,0);

				self.pz2.weights[i].append(0); 
				self.pz2.weights[i].append(0);
				self.pz2.weights[i].append(0); 

			#self.pz2.bias.insert(2,0);
			#self.pz2.bias.append(0); 

			#Robot3, observes relative to robot 2
			self.pz3 = Softmax(); 
			self.pz3.buildGeneralModel(dims=4,numClasses=numClasses,boundries=boundries,B=B,steepness=steep); 

			#add zeros for 0,3,4,5,8,9
			for i in range(0,len(self.pz3.weights)):
				self.pz3.weights[i].insert(0,0);
				self.pz3.weights[i].insert(3,0); 
				self.pz3.weights[i].insert(4,0);
				self.pz3.weights[i].insert(5,0); 
				self.pz3.weights[i].append(0);  
				self.pz3.weights[i].append(0);  
			
			#Robot4, observes relative to robot 3
			self.pz4 = Softmax(); 
			self.pz4.buildGeneralModel(dims=4,numClasses=numClasses,boundries=boundries,B=B,steepness=steep); 

			#zeros at 0,1,4,5,6,9
			for i in range(0,len(self.pz4.weights)):
				self.pz4.weights[i].insert(0,0);
				self.pz4.weights[i].insert(1,0); 
				self.pz4.weights[i].insert(4,0);
				self.pz4.weights[i].insert(5,0);
				self.pz4.weights[i].insert(6,0);
				self.pz4.weights[i].append(0);
	 

			#Robot4, observes relative to robot 3
			self.pz5 = Softmax(); 
			self.pz5.buildGeneralModel(dims=4,numClasses=numClasses,boundries=boundries,B=B,steepness=steep); 

			#zeros at 0,1,2,5,6,7
			for i in range(0,len(self.pz5.weights)):
				self.pz5.weights[i].insert(0,0);
				self.pz5.weights[i].insert(1,0); 
				self.pz5.weights[i].insert(2,0);
				self.pz5.weights[i].insert(5,0);
				self.pz5.weights[i].insert(6,0);
				self.pz5.weights[i].insert(7,0);

			np.save("../models/obs/"+ self.fileNamePrefix + "OBS1.npy",self.pz1);
			np.save("../models/obs/"+ self.fileNamePrefix + "OBS2.npy",self.pz2);
			np.save("../models/obs/"+ self.fileNamePrefix + "OBS3.npy",self.pz3);
			np.save("../models/obs/"+ self.fileNamePrefix + "OBS4.npy",self.pz4);
			np.save("../models/obs/"+ self.fileNamePrefix + "OBS5.npy",self.pz5);
		else:
			self.pz1 = np.load("../models/obs/"+ self.fileNamePrefix + "OBS1.npy").tolist(); 
			self.pz2 = np.load("../models/obs/"+ self.fileNamePrefix + "OBS2.npy").tolist(); 
			self.pz3 = np.load("../models/obs/"+ self.fileNamePrefix + "OBS3.npy").tolist(); 
			self.pz4 = np.load("../models/obs/"+ self.fileNamePrefix + "OBS4.npy").tolist(); 
			self.pz5 = np.load("../models/obs/"+ self.fileNamePrefix + "OBS5.npy").tolist(); 


	
	def buildReward(self,gen = True):
		if(gen): 

			self.r = [0]*len(self.delA);
			
			#square of desired poses with robot 0 in the middle
			#might need to mess with this for observability

			#Just make every action give the same reward
			#goal = [5,5,2.5,7.5,7.5,7.5,7.5,2.5,2.5,2.5]; 
			goal = [-2,0,2,2,2,-4,-1,1,-2,-3]; 

			for i in range(0,len(self.r)):
				self.r[i] = GM();  
				
				self.r[i].addG(Gaussian((np.array(goal)-np.array(self.delA)).tolist(),(np.identity(10)*1).tolist(),1))

			# self.r[0].display(); 
			
			np.save("../models/rew/"+ self.fileNamePrefix + "REW.npy",self.r);

		else:
			self.r = np.load("../models/rew/"+ self.fileNamePrefix + "REW.npy").tolist();




def plot2D(soft,dimsToCut = [2,3], fixedPoints = [0,0], low = [0,0],high = [5,5],vis = True,delta=0.1):
	x, y = np.mgrid[low[0]:high[0]:delta, low[1]:high[1]:delta]
	pos = np.dstack((x, y))  
	resx = int((high[0]-low[0])//delta)+1;
	resy = int((high[1]-low[1])//delta)+1; 

	model = [[[0 for i in range(0,resy)] for j in range(0,resx)] for k in range(0,len(soft.weights))];
	
	xydim = [0,1,2,3]; 
	for d in dimsToCut:
		xydim.remove(d); 

	#print(xydim); 

	for m in range(0,len(soft.weights)):
		for i in range(0,resx):
			xx = (i*(high[0]-low[0])/resx + low[0]);
			for j in range(0,resy):
				yy = (j*(high[1]-low[1])/resy + low[1])
				dem = 0; 
				for k in range(0,len(soft.weights)):
					dem+=np.exp(soft.weights[k][xydim[0]]*xx + soft.weights[k][xydim[1]]*yy + soft.weights[k][dimsToCut[0]]*fixedPoints[0] + soft.weights[k][dimsToCut[1]]*fixedPoints[1] + soft.bias[k]);
				model[m][i][j] = np.exp(soft.weights[m][xydim[0]]*xx + soft.weights[m][xydim[1]]*yy + soft.weights[m][dimsToCut[0]]*fixedPoints[0] + soft.weights[m][dimsToCut[1]]*fixedPoints[1] + soft.bias[m])/dem;

	dom = [[0 for i in range(0,resy)] for j in range(0,resx)]; 
	for m in range(0,len(soft.weights)):
		for i in range(0,resx):
			for j in range(0,resy):
				dom[i][j] = np.argmax([model[h][i][j] for h in range(0,len(soft.weights))]); 
	if(vis):
		plt.contourf(x,y,dom,cmap = 'viridis'); 
		
		fig = plt.figure()
		ax = fig.gca(projection='3d');
		colors = ['b','g','r','c','m','y','k','w','b','g']; 
		for i in range(0,len(model)):
			ax.plot_surface(x,y,model[i],color = colors[i]); 
		plt.title('Softmax Classes')
		plt.show(); 
	else:
		return x,y,dom;




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
	numClasses = 9; 
	boundries = []; 
	for i in range(1,numClasses):
		boundries.append([i,0]); 
	#B = np.matrix(np.concatenate(([-1,1,0,0,-1],[1,-1,0,0,-1],[-1,1,0,0,-1],[1,-1,0,0,-1],[0,0,1,0,-1],[0,0,-1,0,-1],[0,0,0,1,-1],[0,0,0,-1,-1]))).T;

	B = np.matrix(np.concatenate(([-1,1,0,0,-1],[1,-1,0,0,-1],[-1,1,0,0,-1],[1,-1,0,0,-1],[0,0,-1,1,-1],[0,0,1,-1,-1],[0,0,-1,1,-1],[0,0,1,-1,-1]))).T;



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


def moveTest10D():
	#Parrallel-Piped, 4D, Tilted along X and Y
	numClasses = 9; 
	boundries = []; 
	for i in range(1,numClasses):
		boundries.append([i,0]); 
	B = np.matrix(np.concatenate(([-1,1,0,0,-1],[1,-1,0,0,-1],[-1,1,0,0,-1],[1,-1,0,0,-1],[0,0,-1,1,-1],[0,0,1,-1,-1],[0,0,-1,1,-1],[0,0,1,-1,-1]))).T;



	#Notes:
	#There's a push and pull between constants on the same dimension
	#Constants need to both be negative on the same dimension
	#The absolute value of the sum of the constants needs to be <= the sum of the absolute values of the normals
	#higher values squeeze against their direction if constants are negative

	pz = Softmax(); 
	pz.buildGeneralModel(dims=4,numClasses=numClasses,boundries=boundries,B=B,steepness=5); 

	robberPose = [-1,3]; 
	[x,y,dom] = plot2D(pz,dimsToCut = [1,3], fixedPoints = robberPose, low = [-5,-5],high = [5,5],vis = False,delta=0.1);
	for i in range(0,len(dom)):
		for j in range(0,len(dom[i])):
			if(dom[i][j] == 3):
				dom[i][j] = 1; 
			elif(dom[i][j] == 8):
				dom[i][j] = 6; 
			elif(dom[i][j]==7):
				dom[i][j] = 5; 
			elif(dom[i][j] == 4):
				dom[i][j] = 2; 
	plt.contourf(x,y,dom); 
	#plt.imshow(dom,origin='lower'); 
	plt.axis('equal')
	plt.axis('tight'); 
	#plt.xlim([-5,5]); 
	#plt.ylim([-5,5])
	plt.scatter(robberPose[0],robberPose[1],c='r',marker='x',s=100); 
	plt.show(); 


def testVBin4D():
	#Parrallel-Piped, 4D, Tilted along X and Y
	numClasses = 9; 
	boundries = []; 
	for i in range(1,numClasses):
		boundries.append([i,0]); 
	#B = np.matrix(np.concatenate(([-1,1,0,0,-1],[1,-1,0,0,-1],[-1,1,0,0,-1],[1,-1,0,0,-1],[0,0,1,0,-1],[0,0,-1,0,-1],[0,0,0,1,-1],[0,0,0,-1,-1]))).T;
	B = np.matrix(np.concatenate(([-1,1,0,0,-1],[1,-1,0,0,-1],[-1,1,0,0,-1],[1,-1,0,0,-1],[0,0,-1,1,-1],[0,0,1,-1,-1],[0,0,-1,1,-1],[0,0,1,-1,-1]))).T;



	#Notes:
	#There's a push and pull between constants on the same dimension
	#Constants need to both be negative on the same dimension
	#The absolute value of the sum of the constants needs to be <= the sum of the absolute values of the normals
	#higher values squeeze against their direction if constants are negative

	pz = Softmax(); 
	pz.buildGeneralModel(dims=4,numClasses=numClasses,boundries=boundries,B=B,steepness=5); 


	prior = GM(); 
	for i in range(-5,1):
		for j in range(-5,1):
			prior.addG(Gaussian([0,i,0,j],np.identity(4).tolist(),1)); 
	prior.normalizeWeights(); 
	print("Belief constructed")

	#So, west
	posterior1 = pz.runVBND(prior,1); 
	posterior3 = pz.runVBND(prior,3); 
	post = GM(); 
	post.addGM(posterior1); 
	post.addGM(posterior3); 

	copGM = GM();
	dims = [0,2]; 
	for g in post.Gs:
		mean = [g.mean[dims[0]],g.mean[dims[1]]];
		var = [[g.var[dims[0]][dims[0]],g.var[dims[0]][dims[1]]],[g.var[dims[1]][dims[0]],g.var[dims[1]][dims[1]]]]
		weight = g.weight;
		copGM.addG(Gaussian(mean,var,weight));

	robGM = GM();
	dims = [1,3]; 
	for g in post.Gs:
		mean = [g.mean[dims[0]],g.mean[dims[1]]];
		var = [[g.var[dims[0]][dims[0]],g.var[dims[0]][dims[1]]],[g.var[dims[1]][dims[0]],g.var[dims[1]][dims[1]]]]
		weight = g.weight;
		robGM.addG(Gaussian(mean,var,weight));

	[x,y,c] = copGM.plot2D(low=[-5,-5],high=[5,5],vis=False); 
	plt.contourf(x,y,c); 
	plt.show(); 

	[x,y,c] = robGM.plot2D(low=[-5,-5],high=[5,5],vis=False); 
	plt.contourf(x,y,c); 
	plt.show(); 




if __name__ == '__main__':
	a = ModelSpec(); 
	a.buildTransition(); 
	a.buildReward(gen = True); 
	a.buildObs(gen = True); 

	#TestSoftmaxIn10D(); 
	#moveTest10D();

	#testVBin4D(); 







