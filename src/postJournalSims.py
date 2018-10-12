'''
***********************************************************
File: testArena.py
Author: Luke Burks

Research code for the 2018 extension of the 2017 CDC Paper
Testing POMDP policies from VB-POMDP,GM-POMDP, on 
robustness to model errors

Specifically a 2x2xN, where N is the number of solvers, 
of whether or not the particular solver knows an NCV model
is in use, and whether or not it actually is


***********************************************************
'''

from __future__ import division

__author__ = "Luke Burks"
__copyright__ = "Copyright 2016, Cohrint"
__credits__ = ["Luke Burks", "Nisar Ahmed"]
__license__ = "GPL"
__version__ = "1.3.4"
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

def makeSeedBeliefs():

	print("Making Beliefs"); 
	allBels = []; 
	for i in range(-3,4):
		for j in range(-3,4):
			for k in range(-1,2):
				for l in range(-1,2):
					b = GM(); 
					b.addG(Gaussian([i*2,j*2,k/4,l/4],np.identity(4).tolist(),1)); 
					b[0].var[2][2] = 0.25; 
					b[0].var[3][3] = 0.25; 
					allBels.append(b); 
	print("Created {} beliefs".format(len(allBels))); 
	print("Saving Beliefs"); 

	allBels2D = deepcopy(allBels); 
	for i in range(0,len(allBels2D)):
		allBels2D[i] = allBels2D[i].slice2DFrom4D(dims=[0,1],vis = False,retGS = True); 

	f = open("../beliefs/"+ 'D4DiffsSoftmax' + "Beliefs0.npy","w"); 
	np.save(f,allBels);
	f = open("../beliefs/"+ 'D4Diffs' + "Beliefs0.npy","w"); 
	np.save(f,allBels);

	f = open("../beliefs/"+ 'D2DiffsSoftmax' + "Beliefs0.npy","w"); 
	np.save(f,allBels2D);
	f = open("../beliefs/"+ 'D2Diffs' + "Beliefs0.npy","w"); 
	np.save(f,allBels2D);


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
	btmp = btmp.kmeansCondensationN(10); 
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
	btmp = btmp.kmeansCondensationN(10);  
	btmp.normalizeWeights();

	return btmp; 


def beliefUpdateSTM(b,a,o,allMod):
	btmp = GM(); 

	#print(len(allMod.pz[0].Gs)); 
	#print(len(b.Gs)); 

	for obs in allMod.pz[o].Gs:
		for bel in b.Gs:
			mu_j = np.matrix(bel.mean).T; 
			mu_o = np.matrix(obs.mean).T; 
			delA = np.matrix(allMod.delA[a]).T; 
			sig_o = np.matrix(obs.var); 
			sig_j = np.matrix(bel.var); 
			delAVar = np.matrix(allMod.delAVar); 


			h = allMod.STM*(np.add(sig_j,allMod.STM.I*delAVar*allMod.STM.T.I)); 


			weight = obs.weight*bel.weight*(1/np.linalg.det(allMod.STM)); 
			weight = weight*mvn.pdf((allMod.STM*mu_j+delA).T.tolist()[0],mu_o.T.tolist()[0],np.add(sig_o,h)); 
			var = (sig_o.I + h.I).I; 
			mean = var*(sig_o.I*mu_o+h.I*(allMod.STM*mu_j+delA)); 
			weight = weight.tolist(); 
			mean = mean.T.tolist()[0]; 
			var = var.tolist();
			 

			btmp.addG(Gaussian(mean,var,weight)); 
	btmp.normalizeWeights(); 
	#print("Condensing"); 
	btmp = btmp.kmeansCondensationN(10); 
	#btmp.condense(maxMix); 
	btmp.normalizeWeights();
	return btmp; 

def beliefUpdateSoftmaxSTM(b,a,o,allMod):

	btmp = GM(); 
	btmp1 = GM(); 
	for j in b.Gs:
		mean = (np.multiply(allMod.STM,np.matrix(j.mean)) + np.matrix(allMod.delA[a])).tolist()[0]; 
		var = allMod.STM*(np.add(np.matrix(j.var),allMod.STM.I*np.matrix(allMod.delAVar)*allMod.STM.T.I)); 
		weight = j.weight*(1/np.linalg.det(allMod.STM)); 
		btmp1.addG(Gaussian(mean,var,weight)); 
	btmp = allMod.pz.runVBND(btmp1,o); 
	
	#btmp.condense(maxMix);
	btmp = btmp.kmeansCondensationN(10);  
	btmp.normalizeWeights();

	return btmp; 


def sim(policy,initBelief,initPose,allModBels,allModActs,numSteps = 20,useSoft=False,MCTS=False,greedy=False):

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
		x = (allModActs.STM * np.matrix(x).T).T.tolist()[0]; 

		#print(act,x); 
		x = np.random.multivariate_normal(np.array(x)+np.array(allModActs.delA[act]),allModActs.delAVar,size =1)[0].tolist();
		

		#bound the movement
		for i in range(0,len(x)):
			x[i] = max(allModActs.bounds[i][0],x[i]); 
			x[i] = min(allModActs.bounds[i][1],x[i]);

		#print("Getting Observation"); 

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
			b = beliefUpdateSoftmaxSTM(b,act,z,allModBels);
		else:
			b = beliefUpdateSTM(b,act,z,allModBels);  

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


def runMultiSim(think,use,simCount=10,simSteps = 100,alphaNum = 2,useSoft=False,MCTS = False,greedy = False):
	#run simulations
	allSimRewards = [];
	allSimAct = []; 
	allSimX = []; 
	allSimXInd = []; 
	allSimB = [];  

	sys.path.append('../models/') 

	if(useSoft):
		policy = np.load("../policies/D2DiffsSoftmaxAlphas"+think+".npy",encoding='latin1');
		modelModule = __import__('D2DiffsSoftmaxModel', globals(), locals(), ['ModelSpec'],0); 
		modelClass = modelModule.ModelSpec;
		modelName = 'D2DiffsSoftmax'
	else:
		policy = np.load("../policies/D2DiffsAlphas"+think+".npy",encoding='latin1');
		modelModule = __import__('D2DiffsModel', globals(), locals(), ['ModelSpec'],0); 
		modelClass = modelModule.ModelSpec;
		modelName = 'D2Diffs'
	
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
				var = np.identity(2)*4; 
				var = var.tolist(); 
				b.addG(Gaussian([i*3,j*3],var,1))
		b.normalizeWeights();


		x = [np.random.random()*20-10,np.random.random()*20-10];

		print("Starting simulation: " + str(count+1) + " of " + str(simCount));
		
		[allB,allX,allXInd,allAct,allReward] = sim(policy,b,x,allModBels,allModActs,numSteps=simSteps,useSoft=useSoft,MCTS = MCTS,greedy=greedy); 
		
		

		
		allSimRewards.append(allReward);
		allSimB.append([allB]); 
		allSimAct.append([allAct]); 
		allSimX.append([allX]); 
		allSimXInd.append([allXInd]); 
		print("Simulation complete. Reward: " + str(allReward[-1])); 

		suma = 0; 
		for i in range(0,len(allSimRewards)):
			suma += allSimRewards[i][-1]/len(allSimRewards); 
		print("Average Reward So Far: {}".format(suma)); 

	
	suma = 0; 
	for i in range(0,len(allSimRewards)):
		suma+=allSimRewards[i][-1]/simCount; 
	print("Average Reward:{}".format(suma)); 

	#save all data
	dataSave = {"Beliefs":allSimB,"States":allSimX,"States(Ind)":allSimXInd,"Actions":allSimAct,'Rewards':allSimRewards};

	if(not os.path.isdir('../results/'+modelName)):
		os.mkdir('../results/'+modelName);

			
	f = '../results/'+modelName+'/' + modelName + '_Data' + str(alphaNum)+ '.npy';
		

	np.save(f,dataSave); 


def signal_handler(signal, frame):
	print(""); 
	s = raw_input("Confirm STOP? (Y/N)"); 
	if(s.upper() == 'Y'):
		sys.exit(-1); 

if __name__ == '__main__':
	#makeSeedBeliefs();

	signal.signal(signal.SIGINT, signal_handler);

	
	think = 'NCP'; 
	use = 'NCP'
	soft = False; 
	greedy = False;
	MCTS = False 

	if(len(sys.argv)>1):
		think = sys.argv[1]; 
		use = sys.argv[2]; 
		if(sys.argv[3] == 'True'):
			soft = True; 
		else:
			soft = False;
		if(sys.argv[4] == 'True'):
			MCTS = True; 
		else:
			MCTS = False;
		if(sys.argv[5] == 'True'):
			greedy = True; 
		else:
			greedy = False; 


	print("Simulating Policy with: Think={}, Use={}, soft={}, MCTS={}, greedy={}".format(think,use,soft,MCTS,greedy)); 
	runMultiSim(think=think,use=use,simCount=20,simSteps=100,alphaNum=4,useSoft=soft,MCTS = MCTS,greedy=greedy);  


	# a = np.load("../policies/D4DiffsAlphas1.npy");
	# b = GM(); 
	# b.addG(Gaussian([5,0,0,0],np.identity(4).tolist(),1));

	# for c in a:
	# 	print(c.action,continuousDot(b,c)) 



	# A = .9; 
	# s = 2; 
	# mu = -2; 
	# sig = 4; 

	# b = GM(); 
	# b.addG(Gaussian(mu,sig,1)); 
	# b2 = GM(); 
	# b2.addG(Gaussian((A**-1)*mu,(A**-1)*sig*(A**-1),1)); 
	
	# tmp1 = b.pointEval(A*s); 
	# tmp2 = b2.pointEval(s)/((1/A)*(1/A))**(-1/2); 
	# #print(((1/A)*(1/A))**(-1/2))
	# # b.display(); 
	# # b2.display()

	# print(tmp1,tmp2,tmp1-tmp2);


	# A = np.matrix([[1,.8],[0,1]]); 
	# s = np.matrix([2,1]).T; 
	# mu = np.matrix([2,3]).T; 
	# sig = np.matrix([[1,0],[0,1]]); 

	# b1 = GM(); 
	# b1.addG(Gaussian(mu.T.tolist()[0],sig.tolist(),1)); 
	# b2 = GM(); 
	# b2.addG(Gaussian((A.I * mu).T.tolist()[0],A.I*sig*A.T.I,1)); 


	# w = 1/np.linalg.det(A)

	# tmp1 = b1.pointEval((A*s).T.tolist()[0]); 

	# tmp2 = b2.pointEval(s.T.tolist()[0])*w; 
	# print(tmp1,tmp2,tmp1-tmp2); 



