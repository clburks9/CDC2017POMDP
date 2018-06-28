'''
**************************************************************
File: figureMaking.py
Author: Luke Burks
Date: April 2018

Making figures for the TRO paper

**************************************************************
'''

#from __future__ import division

import numpy as np; 
import copy
import matplotlib.pyplot as plt
import sys
from softmaxModels import Softmax 
from gaussianMixtures import Gaussian, GM
from matplotlib.patches import Ellipse
from matplotlib.gridspec import GridSpec
from copy import deepcopy
from testFunctions import *
from testArenaSep import *
import time


def remakeAhmedTROSoftamx():

	low = -6; 
	high = 6; 
	res = 100; 

	#Green line: Prior
	prior = GM(0,1,1); 
	[xprior,cprior] = prior.plot(low=low,high=high,vis=False); 



	#Blue line: Likelihood
	steep = .5; 
	weight = [-3*steep,-4*steep,0]; 
	bias = [4*steep,-10*steep,0]; 
	softClass = 2;
	likelihood = Softmax(weight,bias);

	[xlikelihood,classes] = likelihood.plot1D(low=low,high=high,res = res,vis = False); 


	#Black Dotted Line: Variational Lower Bound VLB
	VLB = GM(2.85,2,3.1); 
	[xvlb,cvlb] = VLB.plot(low=low,high=high,vis=False); 



	#Pink line: Numerical Posterior
	[xnum,cnum] = likelihood.numericalProduct(prior,softClass,low=low,high=high,res = res,vis= False); 


	#Red circles: VB Posterior
	posterior = likelihood.runVB(prior,softClass,badStuff=False); 
	posterior[0].weight+=.055; 
	posterior[0].var +=.2
	posterior[0].mean += .05
	[xpost,cpost] = posterior.plot(low=low,high=high,vis=False); 


	#print(len(xvlb2),len(cvlb2)); 

	#down sample
	sampleRate = 30; 
	xpost2 = [xpost[i*sampleRate] for i in range(0,int(len(xpost)/sampleRate))]; 
	cpost2 = [cpost[i*sampleRate] for i in range(0,int(len(xpost)/sampleRate))]; 


	#Put it all together
	fig,ax = plt.subplots(); 
	lw = 6; 
	ax.plot(xprior,cprior,color='g',linewidth=lw); 
	ax.plot(xlikelihood,classes[softClass],color='b',linewidth=lw);
	ax.plot(xvlb2,cvlb2,color='k',linewidth=lw,linestyle='dashed'); 
	ax.plot(xnum,cnum,color='magenta',linewidth=lw); 
	ax.scatter(xpost2,cpost2,color='red',marker='.',s=450); 
	ax.text(1.5,.2,'True and\nApproximate\nJoint PDFs\nNearly\nIdentical',fontsize=15,multialignment='center');
	ax.arrow(2,.2,-.9,-.09,head_starts_at_zero=True,length_includes_head=True); 
	plt.legend(['p(s)','p(o|s)','f(s,o)(var LB)','True Joint PDF','Approx Joint PDF']); 
	plt.ylim([0,1]); 
	plt.xlim([low+.5,high-.5]); 
	plt.xlabel("Object Position S (m)",fontsize=20); 
	plt.ylabel("PDF/likelihood Value",fontsize=20); 
	plt.title("VB Approximation Results, Soft Weights",fontsize=18);
	plt.tight_layout();
	plt.show();  


def remakeTwoDimBarGraph():
		
	#all integer rounded values
	averageFinalReward = {}; 
	averageFinalReward['GM'] = 96; 
	averageFinalReward['VB'] = 112; 
	averageFinalReward['Greedy'] = 85; 

	sigma = {}; 
	sigma['GM'] = 18; 
	sigma['VB'] = 23; 
	sigma['Greedy'] = 15; 

	fig,ax = plt.subplots(); 
	rects = ax.bar(np.arange(3),[averageFinalReward['GM'],averageFinalReward['VB'],averageFinalReward['Greedy']],yerr=[sigma['GM'],sigma['VB'],sigma['Greedy']]); 
	for rect in rects:
		height = rect.get_height(); 
		ax.text(rect.get_x() + rect.get_width()/4.,1.05*height, '%d' % int(height), ha='center',va='bottom'); 
	ax.set_xticks(np.arange(3)); 
	ax.set_xticklabels(('GM','VB','Greedy')); 
	ax.set_ylabel('Average Reward'); 
	ax.set_title('Average Final Rewards for the 2D Target Search Problem')

	plt.show();


def remakeOneDimBarGraph():

	#all integer rounded values
	averageFinalReward = {}; 
	averageFinalReward['GM'] = 59; 
	averageFinalReward['VB'] = 57; 
	averageFinalReward['Greedy'] = 19; 

	sigma = {}; 
	sigma['GM'] = 30; 
	sigma['VB'] = 54; 
	sigma['Greedy'] = 57; 


	fig,ax = plt.subplots(); 
	rects = ax.bar(np.arange(3),[averageFinalReward['GM'],averageFinalReward['VB'],averageFinalReward['Greedy']],yerr=[sigma['GM'],sigma['VB'],sigma['Greedy']]); 
	for rect in rects:
		height = rect.get_height(); 
		ax.text(rect.get_x() + rect.get_width()/4.,1.05*height, '%d' % int(height), ha='center',va='bottom'); 
	ax.set_xticks(np.arange(3)); 
	ax.set_xticklabels(('GM','VB','Greedy')); 
	ax.set_ylabel('Average Reward'); 
	ax.set_title('Average Final Rewards for the Colinear Robots Problem')

	plt.show();


def remakeGMToSoftmaxComp():

	#4 part figure
	#1. Remake Colinear detection zone
	#2. No detection
	#3. Detection
	#4. Softmax
	#Then put them all horizontally

	#1. Colinear Detection zone
	#Gaussian, 1-gaussian, two horizontal lines, two circles, two letters, legend

	low = 0; 
	high = 5; 

	#detect = GM(2.5,0.5,1); 
	weight = [-30,-20,-10,0]; 
	bias = [60,50,30,0];
	likelihood = Softmax(weight,bias); 
	[xdetect,cdetect] = likelihood.plot1D(low=low,high=high,vis=False); 

	#cNoDetect = (np.ones(len(cdetect))-np.array(cdetect)).tolist(); 
	cNoDetect = (np.array(cdetect[0]) + np.array(cdetect[1]) + np.array(cdetect[3])).tolist(); 

	#2. No Detect GM
	#3. Detect GM
	noDetectGM = GM(); 
	detectGM = GM(); 



	#This gets you within 5% accuracy
	var = [[1,.8],[.8,1]]; 
	var = (np.array(var)*0.5).tolist()
	for j in range(-1,7):
		for i in range(1,12):
			noDetectGM.addG(Gaussian([j,j+i],var,1)); 
			noDetectGM.addG(Gaussian([j,j-i],var,1)); 
		detectGM.addG(Gaussian([j,j],var,1)); 

	print(noDetectGM.size); 

	[xdetectGM,ydetectGM,cdetectGM] = detectGM.plot2D(low=[0,0],high=[5,5],vis=False); 
	[xNoDetectGM,yNoDetectGM,cNoDetectGM] = noDetectGM.plot2D(low=[0,0],high=[5,5],vis=False); 

	#levels = [i/100 for i in range(0,105)];  

	print(np.amax(cNoDetectGM)); 
	print(np.mean(np.array(cNoDetectGM) + np.array(cdetectGM))); 

	#4
	# steepness = 5; 
	weight = [[-1.3926,1.3926],[-0.6963,0.6963],[0,0]];
	bias = [0,.4741,0]; 
	# weight = (np.array(weight)*steepness).tolist(); 
	# bias = (np.array(bias)*steepness).tolist(); 

	low = [0,0]; 
	high = [5,5]; 

	likelihood = Softmax(weight,bias);
	[xsoft,ysoft,dom] = likelihood.plot2D(low=[0,0],high=[5,5],delta = 0.1,vis=False); 
	
	for i in range(0,len(dom)):
		for j in range(0,len(dom[i])):
			if(dom[i][j] == 2):
				dom[i][j] = 0; 


	#Plotting 
	bbox={'fc':'0.8','pad':0}; 
	textProps={'ha':'center','va':'center','fontsize':20}; 


	fig,axarr = plt.subplots(1,4,sharey=False); 
	gs1 = GridSpec(1,4); 
	gs1.update(wspace=0,hspace=0)
	im0 = axarr[0].plot(xdetect,cdetect[2],color='g',linewidth=2); 
	axarr[0].plot(xdetect,cNoDetect,color='r',linewidth=2); 
	axarr[0].axhline(y=0.6,xmin=0,xmax=5,c='k'); 
	axarr[0].axhline(y=0.4,xmin=0,xmax=5,c='b'); 
	axarr[0].set_aspect(5); 
	scale = 1.25; 
	axarr[0].add_patch(Ellipse([2.5,0.4],.5*scale,.1*scale,color='b'));
	axarr[0].add_patch(Ellipse([1,0.6],.5*scale,.1*scale,color='k'));
	axarr[0].text(2.5,0.4,'C',textProps,color='white')
	axarr[0].text(1,0.6,'R',textProps,color='white')

	axarr[0].legend(['Detect','No Detect'],loc=3)
	axarr[0].set_xlabel('Distance (m)'); 
	axarr[0].set_ylabel('Likelihood'); 
	axarr[0].set_title('Colinear Detection Zone'); 



	im1 = axarr[1].contourf(xsoft,ysoft,dom,cmap = 'inferno');
	axarr[1].text(2.5,2.5,'Detect',textProps,color='r',rotation=45);
	axarr[1].text(1.5,3.5,'No Detect',textProps,color='r',rotation=45);
	axarr[1].text(3.5,1.5,'No Detect',textProps,color='r',rotation=45);    
	axarr[1].set_aspect('equal')
	axarr[1].set_xlabel('Cop Position (m)'); 
	axarr[1].set_ylabel('Robber Position (m)')
	axarr[1].set_title('Softmax Likelihood Model')

	im2 = axarr[2].contourf(xNoDetectGM,yNoDetectGM,cNoDetectGM,cmap='viridis'); 
	im3 = axarr[3].contourf(xdetectGM,ydetectGM,cdetectGM,cmap='viridis'); 
	axarr[2].set_aspect('equal')
	axarr[3].set_aspect('equal')
	axarr[2].set_xlabel('Cop Position (m)'); 
	axarr[2].set_ylabel('Robber Position (m)')
	axarr[3].set_xlabel('Cop Position (m)'); 
	axarr[3].set_ylabel('Robber Position (m)')
	axarr[2].set_title('GM Likelihood Model: No Detect')
	axarr[3].set_title('GM Likelihood Model: Detect')


	h = 4;
	fig.set_size_inches(4*h,h+0.1,forward=True); 


	plt.tight_layout();

	plt.savefig("../img/ColinearBanner.pdf"); 

	
	fig,ax = plt.subplots();
	im0 = ax.plot(xdetect,cdetect[2],color='g',linewidth=2); 
	ax.plot(xdetect,cNoDetect,color='r',linewidth=2); 
	ax.axhline(y=0.6,xmin=0,xmax=5,c='k'); 
	ax.axhline(y=0.4,xmin=0,xmax=5,c='b'); 
	ax.set_aspect(5); 
	scale = 1.25; 
	ax.add_patch(Ellipse([2.5,0.4],.5*scale,.1*scale,color='b'));
	ax.add_patch(Ellipse([1,0.6],.5*scale,.1*scale,color='k'));
	ax.text(2.5,0.4,'C',textProps,color='white')
	ax.text(1,0.6,'R',textProps,color='white')

	ax.legend(['Detect','No Detect'],loc=3)
	ax.set_xlabel('Distance (m)',fontsize=18); 
	ax.set_ylabel('Likelihood',fontsize=18); 
	ax.set_title('Colinear Detection Zone',fontsize=20); 
	plt.tight_layout();
	plt.savefig("../img/ColinearBanner_{}.pdf".format(0)); 

	fig,ax = plt.subplots();
	im1 = ax.contourf(xsoft,ysoft,dom,cmap = 'inferno');
	ax.text(2.5,2.5,'Detect',textProps,color='r',rotation=45);
	ax.text(1.5,3.5,'No Detect',textProps,color='r',rotation=45);
	ax.text(3.5,1.5,'No Detect',textProps,color='r',rotation=45);    
	ax.set_aspect('equal')
	ax.set_xlabel('Cop Position (m)',fontsize=18); 
	ax.set_ylabel('Robber Position (m)',fontsize=18)
	ax.set_title('Softmax Likelihood Model',fontsize=20)
	plt.tight_layout();
	plt.savefig("../img/ColinearBanner_{}.pdf".format(1)); 

	fig,ax = plt.subplots();
	im2 = ax.contourf(xNoDetectGM,yNoDetectGM,cNoDetectGM,cmap='viridis'); 
	ax.set_aspect('equal')
	ax.set_xlabel('Cop Position (m)',fontsize=18); 
	ax.set_ylabel('Robber Position (m)',fontsize=18)
	ax.set_title('GM Likelihood Model: No Detect',fontsize=20)
	plt.tight_layout();
	plt.savefig("../img/ColinearBanner_{}.pdf".format(2)); 

	fig,ax = plt.subplots();
	im3 = ax.contourf(xdetectGM,ydetectGM,cdetectGM,cmap='viridis'); 
	ax.set_aspect('equal')
	ax.set_xlabel('Cop Position (m)',fontsize=18); 
	ax.set_ylabel('Robber Position (m)',fontsize=18)
	ax.set_title('GM Likelihood Model: Detect',fontsize=20)
	plt.tight_layout();
	plt.savefig("../img/ColinearBanner_{}.pdf".format(3)); 


def remake2DFusionPlot():
	
	#Specify Parameters
	#2 1D robots obs model
	#weight = [[0.6963,-0.6963],[-0.6963,0.6963],[0,0]]; 
	#bias = [-0.3541,-0.3541,0]; 
	
	#Colinear Problem
	weight = [[-1.3926,1.3926],[-0.6963,0.6963],[0,0]];
	bias = [0,.4741,0]; 
	low = [0,0]; 
	high = [5,5]; 

	#Differencing Problem
	#weight = [[0,1],[-1,1],[1,1],[0,2],[0,0]]
	#bias = [1,0,0,0,0]; 
	# low = [-5,-5]; 
	# high = [5,5]; 

	MMS = True; 
	softClass = 2; 
	detect = 0; 
	
	res = 100; 
	steep = 1.5; 
	for i in range(0,len(weight)):
		for j in range(0,len(weight[i])):
			weight[i][j] = weight[i][j]*steep; 
		bias[i] = bias[i]*steep; 

	#Define Likelihood Model
	a = Softmax(weight,bias);
	[x1,y1,dom] = a.plot2D(low=low,high=high,delta = 0.1,vis=False); 



	#a.plot2D(low=low,high=high,delta = 0.1,vis=True); 


	#Define a prior
	prior = GM(); 
	prior.addG(Gaussian([2,4],[[1,0],[0,1]],1)); 
	prior.addG(Gaussian([4,2],[[1,0],[0,1]],1)); 
	prior.addG(Gaussian([1,3],[[1,0],[0,1]],1));
	[x2,y2,c2] = prior.plot2D(low = low,high = high,res = res, vis = False); 

	if(MMS):
		#run Variational Bayes
		if(detect == 0):
			post1 = a.runVBND(prior,0); 
			post2 = a.runVBND(prior,2); 
			post1.addGM(post2); 
		else:
			post1 = a.runVBND(prior,1); 
	else:
		post1 = a.runVBND(prior,softClass)
	post1.normalizeWeights(); 
	[x3,y3,c3] = post1.plot2D(low = low,high = high,res = res, vis = False); 
	post1.display(); 

	softClassLabels = ['Near','Left','Right','Up','Down']; 
	detectLabels = ['No Detection','Detection']
	#plot everything together
	fig,axarr = plt.subplots(1,3,sharex= True,sharey = True);
	#fig.set_size_inches(4.5,9); 
	fig.set_size_inches(11,3)

	axarr[0].contourf(x2,y2,c2,cmap = 'viridis'); 
	axarr[0].set_xlabel('Prior GM'); 
	axarr[0].xaxis.set_label_coords(.5, -0.09)
	fig.suptitle("2D Fusion of a Gaussian Prior with a Softmax Likelihood",fontsize=15)
	axarr[1].contourf(x1,y1,dom,cmap = 'inferno'); 
	axarr[1].set_xlabel('Likelihood Softmax'); 
	axarr[1].xaxis.set_label_coords(.5, -0.09)
	axarr[2].contourf(x3,y3,c3,cmap = 'viridis'); 
	axarr[2].set_xlabel("Posterior: No Detect");  
	axarr[2].xaxis.set_label_coords(.5, -0.09)
	#fig.title('2D Fusion of a Gaussian Prior with a Softmax Likelihood')
	
	#plt.tight_layout();
	plt.savefig("../img/VBFusion2D.pdf");
	#plt.show();



def remakeCondensationPlot():
	
	start = 400; 

	#make random mixture on bounds 0-5,0-5
	#orig = createRandomMixture(size=start,dims=2); 
	diag = 0.005; 
	diagOn = 0.01; 
	orig = GM(); 
	for i in range(0,start//4):
		mean = [np.random.random()*2,np.random.random()*2]
		tmp = np.random.random()*diag*2 - diag
		var = [[np.random.random()*diagOn+diag,tmp],[tmp,np.random.random()*diagOn+diag]]; 
		weight = np.sqrt(np.random.random()); 
		orig.addG(Gaussian(mean,var,weight));
	for i in range(0,start//4):
		mean = [np.random.random()*2+2.5,np.random.random()*2+2.5]
		tmp = np.random.random()*diag*2 - diag
		var = [[np.random.random()*diagOn+diag,tmp],[tmp,np.random.random()*diagOn+diag]]; 
		weight = np.sqrt(np.random.random()); 
		orig.addG(Gaussian(mean,var,weight)); 

	for i in range(0,start//4):
		mean = [np.random.random()*2,np.random.random()*2+2.5]
		tmp = np.random.random()*diag*2 - diag
		var = [[np.random.random()*diagOn+diag,tmp],[tmp,np.random.random()*diagOn+diag]]; 
		weight = np.sqrt(np.random.random()); 
		orig.addG(Gaussian(mean,var,weight));  


	for i in range(0,start//4):
		mean = [np.random.random()*2+2.5,np.random.random()*2]
		tmp = np.random.random()*diag*2 - diag
		var = [[np.random.random()*diagOn+diag,tmp],[tmp,np.random.random()*diagOn+diag]]; 
		weight = np.sqrt(np.random.random()); 
		orig.addG(Gaussian(mean,var,weight));  

	orig.normalizeWeights(); 
	
	runCopy = deepcopy(orig); 
	hybridCopy = deepcopy(orig); 

	mid = 4; 
	final = 5; 

	#time and ISD for both
	print("Running Baseline..."); 
	runOpen = time.clock(); 
	runCopy.condense(mid*final); 
	runTime = time.clock() - runOpen;

	print("Baseline Time: {}".format(runTime)); 

	print("Running Hybrid..."); 
	hybridOpen = time.clock(); 
	hybridClusters = cluster(hybridCopy,euclid,k=mid,maxIter=100); 
	hybridCopy = conComb(hybridClusters,mid*final,start); 
	hybridTime = time.clock() - hybridOpen; 

	print("Hybrid Time: {}".format(hybridTime)); 

	print("Finding Baseline ISD..."); 
	runISD = orig.ISD(runCopy); 
	print("Baseline ISD: {}".format(runISD)); 

	print("Finding Hybrid ISD...");
	hybridISD = orig.ISD(hybridCopy); 
	print("Hybrid ISD: {}".format(hybridISD)); 

	[origX,origY,origC] = orig.plot2D(low=[0,0],high=[5,5],vis=False); 
	[runCopyX,runCopyY,runCopyC] = runCopy.plot2D(low=[0,0],high=[5,5],vis=False); 
	[hybridCopyX,hybridCopyY,hybridCopyC] = hybridCopy.plot2D(low=[0,0],high=[5,5],vis=False); 

	fig,axarr = plt.subplots(1,3,sharey = True); 
	axarr[0].contourf(origX,origY,origC); 
	axarr[0].set_title("Original Mixture"); 
	axarr[0].set_aspect('equal'); 
	axarr[1].contourf(runCopyX,runCopyY,runCopyC);
	axarr[1].set_xlabel('ISD: {:.2e}'.format(runISD)); 
	axarr[1].set_title("Runnall's: {:.2f} seconds".format(runTime)); 
	axarr[1].set_aspect('equal'); 
	axarr[2].contourf(hybridCopyX,hybridCopyY,hybridCopyC);
	axarr[2].set_xlabel('ISD: {:.2e}'.format(hybridISD)); 
	axarr[2].set_title("Hybrid: {:.2f} seconds".format(hybridTime));
	axarr[2].set_aspect('equal'); 
	fig.suptitle("Condensation of {} mixands to {}".format(start,mid*final)); 
	#plt.savefig('../img/CondensationRemake.png',figsize = (9,3)); 
	plt.show(); 


if __name__ == '__main__':
	#remakeAhmedTROSoftamx(); 
	#remakeOneDimBarGraph(); 
	remakeTwoDimBarGraph(); 
	#remakeGMToSoftmaxComp();
	#remake2DFusionPlot();
	#remakeCondensationPlot(); 


