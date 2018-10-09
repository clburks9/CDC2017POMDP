'''
***********************************************************
File: postJournalSims.py
Author: Luke Burks


***********************************************************
'''

from __future__ import division

__author__ = "Luke Burks"
__copyright__ = "Copyright 2018, Cohrint"
__credits__ = ["Luke Burks", "Nisar Ahmed"]
__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Luke Burks"
__email__ = "luke.burks@colorado.edu"
__status__ = "Development"



from gaussianMixtures import GM,Gaussian
from softmaxModels import Softmax
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn
import os
import sys, getopt
import signal



def makeSeedBeliefs(useSoft):

	# print("Making Beliefs"); 
	# allBels = []; 
	# for i in range(-3,4):
	# 	for j in range(-3,4):
	# 		for k in range(-1,2):
	# 			for l in range(-1,2):
	# 				b = GM(); 
	# 				b.addG(Gaussian([i*2,j*2,k/4,l/4],np.identity(4).tolist(),1)); 
	# 				b[0].var[2][2] = 0.25; 
	# 				b[0].var[3][3] = 0.25; 
	# 				allBels.append(b); 
	# print("Created {} beliefs".format(len(allBels))); 
	# print("Saving Beliefs"); 

	# allBels2D = deepcopy(allBels); 
	# for i in range(0,len(allBels2D)):
	# 	allBels2D[i] = allBels2D[i].slice2DFrom4D(dims=[0,1],vis = False,retGS = True); 

	# f = open("../beliefs/"+ 'D4DiffsSoftmax' + "Beliefs0.npy","w"); 
	# np.save(f,allBels);
	# f = open("../beliefs/"+ 'D4Diffs' + "Beliefs0.npy","w"); 
	# np.save(f,allBels);

	# f = open("../beliefs/"+ 'D2DiffsSoftmax' + "Beliefs0.npy","w"); 
	# np.save(f,allBels2D);
	# f = open("../beliefs/"+ 'D2Diffs' + "Beliefs0.npy","w"); 
	# np.save(f,allBels2D);

	sys.path.append('../models')

	if(useSoft):
		modelModule = __import__('D2Diffs_2SoftmaxModel', globals(), locals(), ['ModelSpec'],0); 
		modelClass = modelModule.ModelSpec;
		modelName = 'D2Diffs_2Softmax'
	else:
		modelModule = __import__('D2Diffs_2Model', globals(), locals(), ['ModelSpec'],0); 
		modelClass = modelModule.ModelSpec;
		modelName = 'D2Diffs_2'

	allModBels = modelClass(); 
	allModBels.buildObs(gen=False);
	allModBels.buildTransition();
	allModBels.buildReward(gen=False); 
	allModActs = modelClass(); 
	allModActs.buildTransition();
	allModActs.buildObs(gen=False); 
	allModActs.buildReward(gen=False);


	print("Making Beliefs"); 
	allBels = []; 

	for trace in range(0,10):
		b=GM(); 
		for i in range(-2,3):
			for j in range(-2,3):
				var = np.identity(2)*6; 
				var = var.tolist(); 
				b.addG(Gaussian([i*3,j*3],var,1))
		b.normalizeWeights(); 
		x = [np.random.random()*20-10,np.random.random()*20-10];
		for step in range(0,10):
			act = getGreedyAction(b);
			x = np.random.multivariate_normal(np.array(x)+np.array(allModActs.delA[act]),allModActs.delAVar,size =1)[0].tolist();
			
			#bound the movement
			for i in range(0,len(x)):
				x[i] = max(allModActs.bounds[i][0],x[i]); 
				x[i] = min(allModActs.bounds[i][1],x[i]);
				
			if(useSoft):
				#Get observation
				ztrial = [0]*allModBels.pz.size; 
				for i in range(0,allModBels.pz.size):
					ztrial[i] = allModBels.pz.pointEvalND(i,x); 
				suma = sum(ztrial); 
				for i in range(0,allModBels.pz.size):
					ztrial[i] = ztrial[i]/suma; 
				z = np.random.choice([0,1,2,3,4],p=ztrial);
			else:
				ztrial = [0]*len(allModBels.pz); 
				for i in range(0,len(allModBels.pz)):
					ztrial[i] = allModBels.pz[i].pointEval(x); 
				suma = sum(ztrial); 
				for i in range(0,len(allModBels.pz)):
					ztrial[i] = ztrial[i]/suma; 
				z = np.random.choice([0,1,2,3,4],p=ztrial);
				
			#print("Updating Belief"); 

			#Belief Update
			if(useSoft):
				b = beliefUpdateSoftmax(b,act,z,allModBels);
			else:
				b = beliefUpdate(b,act,z,allModBels);  

			allBels.append(b); 
	if(useSoft):
		f = "../beliefs/"+ 'D2Diffs_2Softmax' + "Beliefs0.npy"; 
	else:
		f = "../beliefs/"+ 'D2Diffs_2' + "Beliefs0.npy"; 
	np.save(f,allBels);
	

def continuousDot(a,b):
	suma = 0;  

	if(isinstance(a,np.ndarray)):
		a = a.tolist(); 
		a = a[0]; 

	if(isinstance(a,list)):
		a = a[0];

	a.clean(); 
	b.clean(); 

	for k in range(0,a.size):
		for l in range(0,b.size):
			suma += a.Gs[k].weight*b.Gs[l].weight*mvn.pdf(b.Gs[l].mean,a.Gs[k].mean, np.matrix(a.Gs[k].var)+np.matrix(b.Gs[l].var)); 
	return suma; 



def getGreedyAction(bel):

	MAP = bel.findMAPN();
	
	if(abs(MAP[0])>abs(MAP[1])):
		if(MAP[0] > 0):
			act = 0; 
		else:
			act = 1; 
	else:
		if(MAP[1] < 0):
			act = 2; 
		else:
			act = 3; 

	return act; 



def getAction(b,policy):
	act = policy[np.argmax([continuousDot(j,b) for j in policy])].action;
	return act; 

def beliefUpdate(b,a,o,allMod):
	btmp = GM(); 

	for obs in allMod.pz[o].Gs:
		for bel in b.Gs:
			sj = np.matrix(bel.mean).T; 
			si = np.matrix(obs.mean).T; 
			delA = np.matrix(allMod.delA[a]).T; 
			sigi = np.matrix(obs.var); 
			sigj = np.matrix(bel.var); 
			delAVar = np.matrix(allMod.delAVar); 

			weight = obs.weight*bel.weight; 
			weight = weight*mvn.pdf((sj+delA).T.tolist()[0],si.T.tolist()[0],np.add(sigi,sigj,delAVar)); 
			var = (sigi.I + (sigj+delAVar).I).I; 
			mean = var*(sigi.I*si + (sigj+delAVar).I*(sj+delA)); 
			weight = weight.tolist(); 
			mean = mean.T.tolist()[0]; 
			var = var.tolist();
			 

			btmp.addG(Gaussian(mean,var,weight)); 
	btmp.normalizeWeights(); 
	btmp = btmp.kmeansCondensationN(5,2); 
	#btmp.condense(maxMix); 
	btmp.normalizeWeights();
	return btmp; 

def beliefUpdateSoftmax(b,a,o,allMod):

	btmp = GM(); 
	btmp1 = GM(); 
	for j in b.Gs:
		mean = (np.matrix(j.mean) + np.matrix(allMod.delA[a])).tolist()[0]; 
		var = (np.matrix(j.var) + np.matrix(allMod.delAVar)).tolist(); 
		weight = j.weight; 
		btmp1.addG(Gaussian(mean,var,weight)); 
	btmp = allMod.pz.runVBND(btmp1,o); 
	
	#btmp.condense(maxMix);
	btmp = btmp.kmeansCondensationN(5,2);  
	btmp.normalizeWeights();

	return btmp; 


def sim(policy,initBelief,initPose,allModBels,allModActs,numSteps = 20,useSoft=False,greedy=False):

	b = initBelief; 
		
	#Setup data gathering 
	x = initPose;
	allX = []; 
	allX.append(x); 
	allXInd = [0]*len(allModActs.delA[0]); 
	for i in range(0,len(allModActs.delA[0])):
		allXInd[i] = [x[i]]; 

	reward = 0; 
	allReward = [0]; 
	allB = []; 
	allB.append(b); 

	allAct = []; 


	#Simulate
	for count in range(0,numSteps): 

		#print("Step {} of {}".format(count+1,numSteps)); 

		#Get Action
		if(greedy):
			act = getGreedyAction(b);
		else:
			act = getAction(b,policy);
		
		#print("Taking Action"); 

		#Take action
		#x = (allModActs.STM * np.matrix(x).T).T.tolist()[0]; 

		#print(act,x); 
		x = np.random.multivariate_normal(np.array(x)+np.array(allModActs.delA[act]),allModActs.delAVar,size =1)[0].tolist();
		

		#bound the movement
		for i in range(0,len(x)):
			x[i] = max(allModActs.bounds[i][0],x[i]); 
			x[i] = min(allModActs.bounds[i][1],x[i]);

		#print("Getting Observation"); 

		# if(useSoft):
		# 	#Get observation
		# 	ztrial = [0]*allModBels.pz.size; 
		# 	for i in range(0,allModBels.pz.size):
		# 		ztrial[i] = allModBels.pz.pointEvalND(i,x); 
		# 	suma = sum(ztrial); 
		# 	for i in range(0,allModBels.pz.size):
		# 		ztrial[i] = ztrial[i]/suma; 
		# 	z = np.random.choice([0,1,2,3,4],p=ztrial);
		# else:
		# 	ztrial = [0]*len(allModBels.pz); 
		# 	for i in range(0,len(allModBels.pz)):
		# 		ztrial[i] = allModBels.pz[i].pointEval(x); 
		# 	suma = sum(ztrial); 
		# 	for i in range(0,len(allModBels.pz)):
		# 		ztrial[i] = ztrial[i]/suma; 
		# 	z = np.random.choice([0,1,2,3,4],p=ztrial);


	
		if(np.sqrt(x[0]**2 + x[1]**2) <= 1):
			z = 0
		else:
			if(abs(x[0]) > abs(x[1])):
				if(x[0] > 0):
					z = 1; 
				else:
					z = 2; 
			else:
				if(x[1]<0):
					z=3;
				else:
					z=4; 


			
		#print("Updating Belief"); 

		#Belief Update
		if(useSoft):
			b = beliefUpdateSoftmax(b,act,z,allModBels);
		else:
			b = beliefUpdate(b,act,z,allModBels);  

		#print("Saving Data"); 

		#save data
		allB.append(b);
		allX.append(x);
		allAct.append(act); 
		for i in range(0,len(x)):
			allXInd[i].append(x[i]);  

		#reward += allMod.r[act].pointEval(x); 
		if(np.sqrt(x[0]**2+x[1]**2) <= 1):
			reward+=5; 
		else:
			reward+=0; 
		allReward.append(reward); 
		

	allAct.append(-1);
	
	#print("Simulation Complete. Accumulated Reward: " + str(reward));  
	return [allB,allX,allXInd,allAct,allReward]; 


def runMultiSim(simCount=10,simSteps = 100,alphaNum = 2,useSoft=False,greedy = False,timeStep = -1):
	#run simulations
	allSimRewards = [];
	allSimAct = []; 
	allSimX = []; 
	allSimXInd = []; 
	allSimB = [];  

	sys.path.append('../models/') 

	if(useSoft):
		if(not greedy):
			if(timeStep==-1):
				policy = np.load("../policies/D2Diffs_2SoftmaxAlphas"+str(alphaNum)+".npy",encoding='latin1');
			else:
				policy = np.load("../policies/D2Diffs_2Softmax/D2Diffs_2SoftmaxAlphas"+str(alphaNum)+"Step"+str(timeStep)+".npy",encoding='latin1');
		else:
			policy = None; 
		modelModule = __import__('D2Diffs_2SoftmaxModel', globals(), locals(), ['ModelSpec'],0); 
		modelClass = modelModule.ModelSpec;
		modelName = 'D2Diffs_2Softmax'
	else:
		if(not greedy):
			if(timeStep==-1):
				policy = np.load("../policies/D2Diffs_2Alphas"+str(alphaNum)+".npy",encoding='latin1');
			else:
				policy = np.load("../policies/D2Diffs_2/D2Diffs_2Alphas"+str(alphaNum)+"Step"+str(timeStep)+".npy",encoding='latin1');
		else:
			policy = None; 
		modelModule = __import__('D2Diffs_2Model', globals(), locals(), ['ModelSpec'],0); 
		modelClass = modelModule.ModelSpec;
		modelName = 'D2Diffs_2'
	
	#Grab Modeling Code
	allModBels = modelClass(); 
	allModBels.buildObs(gen=False);
	allModBels.buildTransition();
	allModBels.buildReward(gen=False); 
	allModActs = modelClass(); 
	allModActs.buildTransition();
	allModActs.buildObs(gen=False); 
	allModActs.buildReward(gen=False);


	for count in range(0,simCount):


		b=GM(); 
		for i in range(-2,3):
			for j in range(-2,3):
				var = np.identity(2)*6; 
				var = var.tolist(); 
				b.addG(Gaussian([i*3,j*3],var,1))
		b.normalizeWeights();


		x = [np.random.random()*20-10,np.random.random()*20-10];

		print("Starting simulation: " + str(count+1) + " of " + str(simCount));
		
		[allB,allX,allXInd,allAct,allReward] = sim(policy,b,x,allModBels,allModActs,numSteps=simSteps,useSoft=useSoft,greedy=greedy); 
		


		allSimRewards.append(allReward);
		allSimB.append([allB]); 
		allSimAct.append([allAct]); 
		allSimX.append([allX]); 
		allSimXInd.append([allXInd]); 
		print("Simulation complete. Reward: " + str(allReward[-1])); 

		av = 0; 
		for i in range(0,len(allSimRewards)):
			av += allSimRewards[i][-1]/len(allSimRewards); 
		print("Average Reward so Far: {}".format(av)); 

	
	suma = 0; 
	for i in range(0,len(allSimRewards)):
		suma+=allSimRewards[i][-1]/simCount; 
	print("Average Reward:{}".format(suma)); 

	#save all data
	dataSave = {"Beliefs":allSimB,"States":allSimX,"States(Ind)":allSimXInd,"Actions":allSimAct,'Rewards':allSimRewards};

	if(not os.path.isdir('../results/'+modelName)):
		os.mkdir('../results/'+modelName);


	if(greedy == True):
		f = '../results/'+modelName+'/'+modelName+"_Data_Greedy"+str(alphaNum)+'.npy'; 
	else:
		if(timeStep == -1):
			f = '../results/'+modelName+'/'+modelName+"_Data"+str(alphaNum)+'.npy'; 
		else:
			f = '../results/'+modelName+'/'+modelName+"_Data"+str(alphaNum)+'_Step'+str(timeStep)+'.npy'; 
	

	np.save(f,dataSave); 


def signal_handler(signal, frame):
	print(""); 
	s = input("Confirm STOP? (Y/N)"); 
	if(s.upper() == 'Y'):
		sys.exit(-1); 



if __name__ == '__main__':
	# makeSeedBeliefs(False);
	# makeSeedBeliefs(True); 

	signal.signal(signal.SIGINT, signal_handler);

	alphaNum = 0
	soft = False; 
	greedy = False;
	timeStep = -1; 


	if(len(sys.argv)>1):
		alphaNum = sys.argv[1]; 
		if(sys.argv[2] == 'True'):
			soft = True; 
		else:
			soft = False;

		if(sys.argv[3] == 'True'):
			greedy = True; 
		else:
			greedy = False; 

		timeStep = int(sys.argv[4]); 



	print("Simulating Policy with: Alpha={}, Soft={}, Greedy={}".format(alphaNum,soft,greedy)); 
	if(timeStep != -1):
		print("Using Policy from Iteration: {}".format(timeStep)); 
	runMultiSim(simCount=20,simSteps=100,alphaNum=alphaNum,useSoft=soft,greedy=greedy,timeStep=timeStep);  


