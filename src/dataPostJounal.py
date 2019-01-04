import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import binom_test as btest
import statsmodels.api as sm
from softmaxModels import Softmax 
from gaussianMixtures import Gaussian, GM
import matplotlib.animation as animation
import os
from matplotlib import rc
from copy import deepcopy
import time
from testFunctions import *
from testArenaSep import *

def getIterationTimes():
	GMtimes = np.load("../results/TRO_Results_Final/D4Diffs/D4Diffs_TimingNCP.npy"); 
	VBtimes = np.load("../results/TRO_Results_Final/D4DiffsSoftmax/D4DiffsSoftmax_TimingNCP.npy"); 

	# CGMtimes = np.load("../results/D4Diffs/D4Diffs_Cond_TimingNCP.npy");
	# CVBtimes = np.load("../results/D4DiffsSoftmax/D4DiffsSoftmax_Cond_TimingNCP.npy"); 



	#Find the half hourly policies
	GMHours = []; 
	VBHours = []; 
	for i in range(1,8*4+1):
		for j in range(0,len(GMtimes)):
			if(GMtimes[j] > i*3600/4):
				GMHours.append(j-1); 
				break; 
		for j in range(0,len(VBtimes)):
			if(VBtimes[j] > i*3600/4):
				VBHours.append(j-1); 
				break; 

	return GMHours,VBHours; 



def getMeanAndSTD(data):
	#Returns stats of a single set of data, arbitrary number of runs

	finals = []; 
	for r in data['Rewards']:
		finals.append(r[-1]); 
	return np.mean(finals),np.std(finals); 


def make8HourPlot(save = True):

	GMHours,VBHours = getIterationTimes(); 

	GMmeans = []; 
	GMFinals = []; 
	GMSTD = []; 

	for i in range(0,len(GMHours)): 
		suma = 0; 
		data = np.load("../results/TRO_Results_Final/D4Diffs/D4Diffs_Data_Step{}.npy".format(GMHours[i]),encoding='latin1').item(); 

		for r in data['Rewards']:
			GMFinals.append(r[-1]); 
			#suma += r[-1]/len(data['Rewards']); 

		GMmeans.append(np.mean(GMFinals)); 
		GMSTD.append(np.std(GMFinals)); 

	x = [i/4 for i in range(1,len(GMmeans))]; 
	plt.plot(x,GMmeans[1:],c='b'); 
	plt.errorbar(x,GMmeans[1:],yerr = GMSTD[1:],c='b'); 

	#VB

	VBmeans = [];
	VBSTD = [];  
	VBFinals = []; 
	for i in range(0,len(VBHours)):

		suma = 0; 
		data = np.load("../results/TRO_Results_Final/D4DiffsSoftmax/D4DiffsSoftmax_Data_Step{}.npy".format(VBHours[i]),encoding='latin1').item(); 

		for r in data['Rewards']:
			VBFinals.append(r[-1]); 

		VBmeans.append(np.mean(VBFinals)); 
		VBSTD.append(np.std(VBFinals));  



	x = [i/4 for i in range(1,len(VBmeans))]; 

	plt.plot(x,VBmeans[1:],c='g'); 
	plt.xlabel("Hours"); 
	plt.ylabel("Average Reward")
	plt.errorbar(np.array(x)+0.025,VBmeans[1:],yerr = VBSTD[1:],c='g'); 


	data = np.load('../results/accumlatedResults.npy').item(); 


	VBFinal = data['VBFinalRewards']; 
	GMFinal = data['GMFinalRewards']; 
	GreedyFinal = data['greedFinalRewards']; 
	PerfectFinal = data['perfectFinalRewards']; 

	# gmean,gstd = getMeanAndSTD(GreedyFinal); 
	# print(gmean,gstd); 


	VBFinalRewards = []; 
	vbr = VBFinal['Rewards']; 
	for r in vbr:
		VBFinalRewards.append(r[-1]); 

	GMFinalRewards = []; 
	gmr = GMFinal['Rewards']; 
	for r in gmr:
		GMFinalRewards.append(r[-1]); 

	greedFinalRewards = []; 
	greedr = GreedyFinal['Rewards']; 
	for r in greedr:
		greedFinalRewards.append(r[-1]); 

	perfectFinalRewards = []; 
	perr = PerfectFinal['Rewards']; 
	for r in perr:
		perfectFinalRewards.append(r[-1]); 

	# print(np.std(VBFinalRewards))
	plt.axhline(np.mean(VBFinalRewards),c='g',linestyle='--');
	plt.axhline(np.mean(GMFinalRewards),c='b',linestyle='--'); 
	plt.axhline(np.mean(greedFinalRewards),c='r',linestyle='--')
	plt.axhline(np.mean(perfectFinalRewards),c='k',linestyle='--')
	# plt.errorbar(0,np.mean(VBFinalRewards),yerr=np.std(VBFinalRewards));  
	plt.title("Comparison of Policies at various solution times")
	if(save):
		plt.savefig('../../../../../mnt/c/Users/clbur/OneDrive/Work Docs/Presentations/11_5_18 Nisar Meeting/LayerPlot.png'); 
	if(show):
		plt.show(); 


def plotCrossStich():

	print("Plotting First 8 Hours"); 
	make8HourPlot(show=False); 

	print("Plotting VB To GM Extra"); 

	VBToGMSplits = [3,4,6,7,8,10,11,12,13,15,16,17,18,19,21,22]; 
	GMToVBSplits = [9,21,32,43,54,66,77,88,99,111,122,133,144,155,167,178]; 

	VBToGMFinals = []; 
	VBToGMMeans= []; 
	VBToGMSTD = []; 

	for i in range(0,len(VBToGMSplits)): 
		suma = 0; 
		data = np.load("../results/TRO_Results_Final/CrossVBToGM/D4DiffsVBToGM_Data_Step{}.npy".format(VBToGMSplits[i]),encoding='latin1').item(); 

		for r in data['Rewards']:
			VBToGMFinals.append(r[-1]); 
			#suma += r[-1]/len(data['Rewards']); 

		VBToGMMeans.append(np.mean(VBToGMFinals)); 
		VBToGMSTD.append(np.std(VBToGMFinals)); 

	x = [i/4 + 8 for i in range(0,len(VBToGMMeans))]; 
	plt.plot(x,VBToGMMeans,c='g'); 
	plt.errorbar(x,VBToGMMeans,yerr = VBToGMSTD,c='g'); 

	print("Plotting GM To VB Extra"); 

	GMToVBFinals = []; 
	GMToVBMeans= []; 
	GMToVBSTD = []; 

	for i in range(0,len(GMToVBSplits)): 
		suma = 0; 
		data = np.load("../results/TRO_Results_Final/CrossGMToVB/D4DiffsGMToVB_Data_Step{}.npy".format(GMToVBSplits[i]),encoding='latin1').item(); 

		for r in data['Rewards']:
			GMToVBFinals.append(r[-1]); 
			#suma += r[-1]/len(data['Rewards']); 

		GMToVBMeans.append(np.mean(GMToVBFinals)); 
		GMToVBSTD.append(np.std(GMToVBFinals)); 

	x = [i/4 + 8 for i in range(0,len(GMToVBMeans))]; 
	plt.plot(x,GMToVBMeans,c='b'); 
	plt.errorbar(x,GMToVBMeans,yerr = GMToVBSTD,c='b'); 

	plt.show(); 


def extractAndSaveAllData():

	GMHours,VBHours = getIterationTimes(); 

	GMmeans = []; 
	GMFinals = []; 
	GMSTD = []; 

	for i in range(0,len(GMHours)): 
		suma = 0; 
		data = np.load("../results/TRO_Results_Final/D4Diffs/D4Diffs_Data_Step{}.npy".format(GMHours[i]),encoding='latin1').item(); 

		for r in data['Rewards']:
			GMFinals.append(r[-1]); 
			#suma += r[-1]/len(data['Rewards']); 

		GMmeans.append(np.mean(GMFinals)); 
		GMSTD.append(np.std(GMFinals)); 


	VBmeans = [];
	VBSTD = [];  
	VBFinals = []; 
	for i in range(0,len(VBHours)):

		suma = 0; 
		data = np.load("../results/TRO_Results_Final/D4DiffsSoftmax/D4DiffsSoftmax_Data_Step{}.npy".format(VBHours[i]),encoding='latin1').item(); 

		for r in data['Rewards']:
			VBFinals.append(r[-1]); 

		VBmeans.append(np.mean(VBFinals)); 
		VBSTD.append(np.std(VBFinals));  


	VBFinal = np.load("../results/TRO_Results_Final/D4DiffsSoftmax/D4DiffsSoftmax_Data4.npy",encoding='latin1').item(); 
	GMFinal = np.load("../results/TRO_Results_Final/D4Diffs/D4Diffs_Data5.npy",encoding='latin1').item(); 
	GreedyFinal = np.load("../results/D4Diffs_Data_Greedy_Stay5.npy",encoding='latin1').item(); 
	PerfectFinal = np.load("../results/D4Diffs/perfectKnowledgeResults_Stay.npy",encoding='latin1').item(); 

	# gmean,gstd = getMeanAndSTD(GreedyFinal); 
	# print(gmean,gstd); 


	VBFinalRewards = []; 
	vbr = VBFinal['Rewards']; 
	for r in vbr:
		VBFinalRewards.append(r[-1]); 

	GMFinalRewards = []; 
	gmr = GMFinal['Rewards']; 
	for r in gmr:
		GMFinalRewards.append(r[-1]); 

	greedFinalRewards = []; 
	greedr = GreedyFinal['Rewards']; 
	for r in greedr:
		greedFinalRewards.append(r[-1]); 

	perfectFinalRewards = []; 
	per = PerfectFinal['Rewards']; 
	for r in per:
		perfectFinalRewards.append(r[-1]); 


	VBLowQFinal = np.load("../results/D4DiffsSoftmax_Data_LowQ.npy",encoding='latin1').item(); 
	GMLowQFinal = np.load("../results/D4Diffs_Data_LowQ.npy",encoding='latin1').item(); 
	GreedyLowQFinal = np.load("../results/D4Diffs_Data_LowQ_Greedy_Stay.npy",encoding='latin1').item(); 
	PerfectLowQFinal = np.load("../results/D4Diffs/perfectKnowledgeResults_LowQ_Stay.npy",encoding='latin1').item(); 


	PerfectLowQ = []; 
	per = PerfectLowQFinal['Rewards']; 
	for r in per:
		PerfectLowQ.append(r[-1]); 


	VBLowQ = []; 
	per = VBLowQFinal['Rewards']; 
	for r in per:
		VBLowQ.append(r[-1]); 

	GMLowQ = []; 
	per = GMLowQFinal['Rewards']; 
	for r in per:
		GMLowQ.append(r[-1]); 

	GreedyLowQ = []; 
	per = GreedyLowQFinal['Rewards']; 
	for r in per:
		GreedyLowQ.append(r[-1]); 

	toSave = {}; 
	toSave['VBFinalRewards'] = VBFinalRewards; 
	toSave['GMFinalRewards'] = GMFinalRewards; 
	toSave['greedFinalRewards'] = greedFinalRewards; 
	toSave['perfectFinalRewards'] = perfectFinalRewards; 
	toSave['VBmeans'] = VBmeans; 
	toSave['VBSTD'] = VBSTD; 
	toSave['GMmeans'] = GMmeans; 
	toSave['GMSTD'] = GMSTD; 

	toSave['VBLowQ'] = VBLowQ; 
	toSave['GMLowQ'] = GMLowQ; 
	toSave['GreedyLowQ'] = GreedyLowQ; 
	toSave['PerfectLowQ'] = PerfectLowQ; 


	np.save('../results/accumlatedResults.npy',toSave); 


def makeLayeredPlot():

	data = np.load('../results/accumlatedResults.npy').item(); 
	VBmeans = data['VBmeans']; 
	VBSTD = data['VBSTD']; 
	GMmeans = data['GMmeans']; 
	GMSTD = data['GMSTD']; 
	VBFinalRewards = data['VBFinalRewards']; 
	GMFinalRewards = data['GMFinalRewards'];
	greedFinalRewards = data['greedFinalRewards']; 
	perfectFinalRewards = data['perfectFinalRewards']; 


	GMHours,VBHours = getIterationTimes(); 

	#plt.rc('text',usetex=True)

	plt.axhline(np.mean(perfectFinalRewards),c='k',linestyle='--',linewidth=2)
	plt.axhline(np.mean(VBFinalRewards),c='g',linestyle='--',linewidth=2);
	plt.axhline(np.mean(GMFinalRewards),c='b',linestyle='--',linewidth=2); 
	plt.axhline(np.mean(greedFinalRewards),c='r',linestyle='--',linewidth=2)
	


	x = [i/4 for i in range(1,len(VBmeans))]; 
	plt.plot(x,VBmeans[1:],c='g',linewidth=3); 
	x = [i/4 for i in range(1,len(GMmeans))]; 
	plt.plot(x,GMmeans[1:],c='b',linewidth=3); 

	plt.legend(['Perfect Information Average','VB Final Mean','GM Final Mean','Greedy Mean','VB Intermediate Mean','GM Intermediate Mean'])

	x = [i/4 for i in range(1,len(VBmeans))]; 
	plt.xlabel("Hours of Policy Approximation"); 
	plt.ylabel("Average Reward")
	plt.plot(x,np.array(VBmeans[1:])+np.array(VBSTD[1:]),linestyle=':',color='g'); 
	plt.plot(x,np.array(VBmeans[1:])-np.array(VBSTD[1:]),linestyle=':',color='g');
	plt.fill_between(x,np.array(VBmeans[1:])+np.array(VBSTD[1:]),np.array(VBmeans[1:])-np.array(VBSTD[1:]),alpha = 0.2,color='g'); 
	#plt.errorbar(np.array(x)+0.05,VBmeans[1:],yerr = VBSTD[1:],c='g'); 


	x = [i/4 for i in range(1,len(GMmeans))]; 
	#plt.plot(x,GMmeans[1:],c='b'); 
	plt.plot(x,np.array(GMmeans[1:])+np.array(GMSTD[1:]),linestyle=':',color='b'); 
	plt.plot(x,np.array(GMmeans[1:])-np.array(GMSTD[1:]),linestyle=':',color='b');
	plt.fill_between(x,np.array(GMmeans[1:])+np.array(GMSTD[1:]),np.array(GMmeans[1:])-np.array(GMSTD[1:]),alpha = 0.15,color='b'); 
	#plt.errorbar(x,GMmeans[1:],yerr = GMSTD[1:],c='b'); 

	
	# plt.errorbar(0,np.mean(VBFinalRewards),yerr=np.std(VBFinalRewards));  
	#plt.title("Comparison of Policies at various solution times")

	

	ax1 = plt.gca(); 
	#ax2 = ax1.twiny(); 
	# labels = []; 
	# for i in range(0,len(VBHours)):
	# 	labels.append(str(VBHours[i])+'/'+str(GMHours[i])); 

	# x = [i/4 + .4 for i in range(0,len(GMmeans)-20)]; 
	# ax2.set_xticks(x); 
	# ax2.set_xticklabels(labels); 

	plt.xlim([.25,3.5]); 
	plt.ylim([98,112])
	#ax2.set_xlim([.25,3.5]);

	plt.show(); 



def makeHighQPlots(save = True):

	data = np.load('../results/accumlatedResults.npy').item(); 


	VB = data['VBFinalRewards']; 
	GM = data['GMFinalRewards']; 
	Greedy = data['greedFinalRewards']; 
	Perfect = data['perfectFinalRewards']; 

	print("Perfect, Mean: {}, STD: {}".format(np.mean(Perfect),np.std(Perfect)))
	print("VB, Mean: {}, STD: {}".format(np.mean(VB),np.std(VB))); 
	print("GM, Mean: {}, STD: {}".format(np.mean(GM),np.std(GM))); 
	print("Greedy, Mean: {}, STD: {}".format(np.mean(Greedy),np.std(Greedy))); 

	# plt.figure();
	# plt.boxplot([Perfect,VB,GM,Greedy],whis=1000,labels=['Perfect Info','VB','GM','Greedy']); 
	# plt.title('High Q 1000-Sim Rewards Distributions'); 
	# #plt.set_aspect('equal'); 
	# if(save):
	# 	plt.savefig('../../../../../mnt/c/Users/clbur/OneDrive/Work Docs/Presentations/11_5_18 Nisar Meeting/HighQBox.png'); 
	# else:
	# 	plt.show(); 

	bins = [i*5+30 for i in range(0,64)]
	

	fig,axarr = plt.subplots(4,1,sharex=True,sharey=True); 
	axarr[1].hist(VB,density=True,bins=bins,color='g'); 
	axarr[1].set_xlabel('VB'); 
	mu,std = norm.fit(VB); 
	xmin,xmax = axarr[1].get_xlim(); 
	x = np.linspace(xmin,xmax,100); 
	p = norm.pdf(x,mu,std); 
	axarr[1].plot(x,p,'k',linewidth=2); 
	axarr[1].set_xlim([0,250])
	axarr[1].set_ylim([0,0.02]); 

	axarr[0].hist(Perfect,density=True,bins=bins,color='k'); 
	axarr[0].set_xlabel('Perfect Info');
	mu,std = norm.fit(Perfect); 
	xmin,xmax = axarr[0].get_xlim(); 
	x = np.linspace(xmin,xmax,100); 
	p = norm.pdf(x,mu,std); 
	axarr[0].plot(x,p,'r',linewidth=2); 
	axarr[0].set_xlim([0,250])
	axarr[0].set_ylim([0,0.02]); 

	axarr[2].hist(GM,density=True,bins=bins,color='b'); 
	axarr[2].set_xlabel('GM');
	mu,std = norm.fit(GM); 
	xmin,xmax = axarr[2].get_xlim(); 
	x = np.linspace(xmin,xmax,100); 
	p = norm.pdf(x,mu,std); 
	axarr[2].plot(x,p,'k',linewidth=2); 
	axarr[2].set_xlim([0,250])
	axarr[2].set_ylim([0,0.02]); 

	axarr[3].hist(Greedy,density=True,bins=bins,color='r'); 
	axarr[3].set_xlabel('Greedy');
	mu,std = norm.fit(Greedy); 
	xmin,xmax = axarr[3].get_xlim(); 
	x = np.linspace(xmin,xmax,100); 
	p = norm.pdf(x,mu,std); 
	axarr[3].plot(x,p,'k',linewidth=2); 
	axarr[3].set_xlim([0,250])
	axarr[3].set_ylim([0,0.02]); 

	axarr[1].axvline(np.mean(VB),c='k'); 
	axarr[0].axvline(np.mean(Perfect),c='r'); 
	axarr[2].axvline(np.mean(GM),c='k'); 
	axarr[3].axvline(np.mean(Greedy),c='k'); 

	fig.suptitle("HighQ 1000-Sim Histograms")
	#plt.set_aspect('equal');
	if(save):
		plt.savefig('../../../../../mnt/c/Users/clbur/OneDrive/Work Docs/Presentations/11_5_18 Nisar Meeting/HighQHists.png'); 
	else:
		plt.show(); 


def makeLowQPlots(save = True):

	data = np.load('../results/accumlatedResults.npy').item(); 


	VBLowQ = data['VBLowQ']; 
	GMLowQ = data['GMLowQ']; 
	GreedyLowQ = data['GreedyLowQ']; 
	PerfectLowQ = data['PerfectLowQ']; 

	print("Perfect, Mean: {}, STD: {}".format(np.mean(PerfectLowQ),np.std(PerfectLowQ)))
	print("VB, Mean: {}, STD: {}".format(np.mean(VBLowQ),np.std(VBLowQ))); 
	print("GM, Mean: {}, STD: {}".format(np.mean(GMLowQ),np.std(GMLowQ))); 
	print("Greedy, Mean: {}, STD: {}".format(np.mean(GreedyLowQ),np.std(GreedyLowQ))); 

	# plt.figure(); 
	
	# plt.boxplot([PerfectLowQ,VBLowQ,GMLowQ,GreedyLowQ],whis=1000,labels=['Perfect Info','VB','GM','Greedy']); 
	# plt.title('LowQ 1000-Sim Rewards Distributions'); 
	# if(save):
	# 	plt.savefig('../../../../../mnt/c/Users/clbur/OneDrive/Work Docs/Presentations/11_5_18 Nisar Meeting/LowQBox.png'); 
	# else:
	# 	plt.show(); 


	bins = [i*5+30 for i in range(0,64)]
	

	fig,axarr = plt.subplots(4,1,sharex=True,sharey=True); 
	axarr[1].hist(VBLowQ,density=True,bins=bins,color='g'); 
	axarr[1].set_xlabel('VB'); 
	mu,std = norm.fit(VBLowQ); 
	xmin,xmax = axarr[1].get_xlim(); 
	x = np.linspace(xmin,xmax,100); 
	p = norm.pdf(x,mu,std); 
	axarr[1].plot(x,p,'k',linewidth=2); 
	axarr[1].set_xlim([0,250])
	axarr[1].set_ylim([0,0.02]); 

	axarr[0].hist(PerfectLowQ,density=True,bins=bins,color='k'); 
	axarr[0].set_xlabel('Perfect Info');
	mu,std = norm.fit(PerfectLowQ); 
	xmin,xmax = axarr[0].get_xlim(); 
	x = np.linspace(xmin,xmax,100); 
	p = norm.pdf(x,mu,std); 
	axarr[0].plot(x,p,'r',linewidth=2); 
	axarr[0].set_xlim([0,250])
	axarr[0].set_ylim([0,0.02]); 

	axarr[2].hist(GMLowQ,density=True,bins=bins,color='b'); 
	axarr[2].set_xlabel('GM');
	mu,std = norm.fit(GMLowQ); 
	xmin,xmax = axarr[2].get_xlim(); 
	x = np.linspace(xmin,xmax,100); 
	p = norm.pdf(x,mu,std); 
	axarr[2].plot(x,p,'k',linewidth=2); 
	axarr[2].set_xlim([0,250])
	axarr[2].set_ylim([0,0.02]); 

	axarr[3].hist(GreedyLowQ,density=True,bins=bins,color='r'); 
	axarr[3].set_xlabel('Greedy');
	mu,std = norm.fit(GreedyLowQ); 
	xmin,xmax = axarr[3].get_xlim(); 
	x = np.linspace(xmin,xmax,100); 
	p = norm.pdf(x,mu,std); 
	axarr[3].plot(x,p,'k',linewidth=2); 
	axarr[3].set_xlim([0,250])
	axarr[3].set_ylim([0,0.02]); 

	axarr[1].axvline(np.mean(VBLowQ),c='k'); 
	axarr[0].axvline(np.mean(PerfectLowQ),c='r'); 
	axarr[2].axvline(np.mean(GMLowQ),c='k'); 
	axarr[3].axvline(np.mean(GreedyLowQ),c='k');

	fig.suptitle("LowQ 1000-Sim Histograms")
	if(save):
		plt.savefig('../../../../../mnt/c/Users/clbur/OneDrive/Work Docs/Presentations/11_5_18 Nisar Meeting/LowQHists.png'); 
	else:
		plt.show(); 



def SDExploration(save=True):

	data = np.load('../results/accumlatedResults.npy').item(); 

	vbf2 = data['VBLowQ']; 
	gmf2 = data['GMLowQ']; 
	gf2 = data['GreedyLowQ']; 
	pf2 = data['PerfectLowQ']; 

	vbf1 = data['VBFinalRewards']; 
	gmf1 = data['GMFinalRewards'];
	gf1 = data['greedFinalRewards']; 
	pf1 = data['perfectFinalRewards']; 

	print(np.mean(pf2)); 
	#HighQ = np.transpose(np.array([[np.mean(pf1),np.std(pf1)],[np.mean(vbf1),np.std(vbf1)],[np.mean(gmf1),np.std(gmf1)],[np.mean(gf1),np.std(gf1)]]))
	#LowQ = [[np.mean(pf2),np.std(pf2)],[np.mean(vbf2),np.std(vbf2)],[np.mean(gmf2),np.std(gmf2)],[np.mean(gf2),np.std(gf2)]]

	HighMean = [np.mean(pf1),np.mean(vbf1),np.mean(gmf1),np.mean(gf1)]; 
	HighSD = [np.std(pf1),np.std(vbf1),np.std(gmf1),np.std(gf1)]; 

	LowMean = [np.mean(pf2),np.mean(vbf2),np.mean(gmf2),np.mean(gf2)]; 
	LowSD = [np.std(pf2),np.std(vbf2),np.std(gmf2),np.std(gf2)]; 

	plt.figure();

	plt.plot(HighMean,HighSD,c='r'); 
	plt.plot(LowMean,LowSD,c='g'); 

	plt.legend(['HighQ','LowQ']); 
	plt.xlabel("Mean"); 
	plt.ylabel("SD"); 
	if(save):
		plt.savefig('../../../../../mnt/c/Users/clbur/OneDrive/Work Docs/Presentations/11_5_18 Nisar Meeting/SDExploration.png'); 
	else:
		plt.show(); 


def actionInvestigation():

	VBFinal = np.load("../results/TRO_Results_Final/D4DiffsSoftmax/D4DiffsSoftmax_Data4.npy",encoding='latin1').item(); 
	GMFinal = np.load("../results/TRO_Results_Final/D4Diffs/D4Diffs_Data5.npy",encoding='latin1').item(); 
	GreedyFinal = np.load("../results/D4Diffs_Data_Greedy_Stay5.npy",encoding='latin1').item(); 
	PerfectFinal = np.load("../results/D4Diffs/perfectKnowledgeResults_Stay.npy",encoding='latin1').item(); 


	VBLowQFinal = np.load("../results/D4DiffsSoftmax_Data_LowQ.npy",encoding='latin1').item(); 
	GMLowQFinal = np.load("../results/D4Diffs_Data_LowQ.npy",encoding='latin1').item(); 
	GreedyLowQFinal = np.load("../results/D4Diffs_Data_LowQ_Greedy_Stay.npy",encoding='latin1').item(); 
	PerfectLowQFinal = np.load("../results/D4Diffs/perfectKnowledgeResults_LowQ_Stay.npy",encoding='latin1').item(); 

	# print(VBFinal.keys()); 
	# print(len(VBFinal['Actions'])); 
	# print(len(VBFinal['States']))

	#make a cloud plot, where each action has a color, and it's plotted at the state of that time

	runs = [PerfectFinal,VBFinal,GMFinal,GreedyFinal,PerfectLowQFinal,VBLowQFinal,GMLowQFinal,GreedyLowQFinal]; 
	names = ["Perfect","VB","GM","Greedy","PerfectLowQ","VBLowQ","GMLowQ","GreedyLowQ"];

	for r in range(0,len(runs)):
		plt.cla(); 
		plt.clf(); 

		print(names[r]); 

		acts = runs[r]['Actions']; 
		states = runs[r]['States']; 

		colors = ['b','r','g','m','y']; 
		allColors = []; 
		for i in range(0,len(acts)): 
			for j in range(1,len(acts[i][0])-1):
				allColors.append(colors[acts[i][0][j]]); 


	#	print(np.array(acts).shape)

		allX = []; 
		allY = []; 
		for i in range(0,len(states)):
			for j in range(1,len(states[i][0])-1):
				#print(states[i][0][j][0])
				allX.append(states[i][0][j][0]); 
				allY.append(states[i][0][j][1]); 

		# print(len(allX)); 
		# print(len(allY));

		plt.axes().set_aspect('equal');
		plt.scatter(allX,allY,color=allColors,s=1);
		plt.xlabel("$\Delta$X") 
		plt.ylabel("$\Delta$Y") 
		plt.title(names[r] + " Actions"); 
		plt.savefig('../img/'+names[r]+"Actions.png");
		plt.savefig('../../../../../mnt/c/Users/clbur/OneDrive/Work Docs/Presentations/11_5_18 Nisar Meeting/'+names[r]+'Actions.png'); 



def actionConfusion():
	VBFinal = np.load("../results/TRO_Results_Final/D4DiffsSoftmax/D4DiffsSoftmax_Data4.npy",encoding='latin1').item(); 
	GMFinal = np.load("../results/TRO_Results_Final/D4Diffs/D4Diffs_Data5.npy",encoding='latin1').item(); 
	GreedyFinal = np.load("../results/D4Diffs_Data_Greedy_Stay5.npy",encoding='latin1').item(); 
	PerfectFinal = np.load("../results/D4Diffs/perfectKnowledgeResults_Stay.npy",encoding='latin1').item(); 


	VBLowQFinal = np.load("../results/D4DiffsSoftmax_Data_LowQ.npy",encoding='latin1').item(); 
	GMLowQFinal = np.load("../results/D4Diffs_Data_LowQ.npy",encoding='latin1').item(); 
	GreedyLowQFinal = np.load("../results/D4Diffs_Data_LowQ_Greedy_Stay.npy",encoding='latin1').item(); 
	PerfectLowQFinal = np.load("../results/D4Diffs/perfectKnowledgeResults_LowQ_Stay.npy",encoding='latin1').item(); 


	

	runs = [PerfectFinal,VBFinal,GMFinal,GreedyFinal,PerfectLowQFinal,VBLowQFinal,GMLowQFinal,GreedyLowQFinal]; 
	names = ["Perfect","VB","GM","Greedy","PerfectLowQ","VBLowQ","GMLowQ","GreedyLowQ"];
	# runs = [VBFinal]; 
	# names = ['VB']; 

	for r in range(0,len(runs)):
		plt.cla(); 
		plt.clf(); 

		print(names[r]); 

		acts = runs[r]['Actions']; 
		states = runs[r]['States']; 

		confusion = np.zeros(shape=(5,5)); 

		for i in range(0,len(states)):
			for j in range(1,len(states[i][0])-1):
				# #print(states[i][0][j][0])
				# allX.append(states[i][0][j][0]); 
				# allY.append(states[i][0][j][1]); 
				x = states[i][0][j]; 
				#Determine Quadrant
				
				if(np.sqrt(x[0]**2 + x[1]**2) < 1):
					first = 4; 
				else:
					if(abs(x[0])>abs(x[1])):
						if(x[0] > 0):
							#action should be left
							first = 0; 
						else:
							#action should be right
							first = 1;
					else:
						if(x[1] > 0):
							first = 3; 
						else:
							first = 2; 
				second = acts[i][0][j];
				confusion[first][second] += 1; 

		confusion += 0.00000000001
		for i in range(0,5):
			if(sum(confusion[i])>0):
				confusion[i] = confusion[i]/sum(confusion[i])
		

		plt.imshow(np.log(confusion),vmin=-10,vmax=0);
		plt.colorbar(); 
		plt.xlabel("Observed Action"); 
		plt.ylabel("Perfect Information Action"); 

		plt.yticks([0,1,2,3,4],['Left','Right','Up','Down','Stay'],rotation=90,va='center'); 
		plt.xticks([0,1,2,3,4],['Left','Right','Up','Down','Stay']); 
		plt.xlim([-0.5,4.5]); 
		plt.ylim([-0.5,4.5]);
		plt.title('Log-Probability of Confusion for {}'.format(names[r]));
		#plt.show(); 
		plt.savefig('../../../../../mnt/c/Users/clbur/OneDrive/Work Docs/Presentations/11_5_18 Nisar Meeting/'+names[r]+'LogConfusion.png'); 

		plt.clf(); 
		plt.cla(); 
		plt.imshow(confusion,vmin=0,vmax=1);
		plt.colorbar(); 
		plt.xlabel("Observed Action"); 
		plt.ylabel("Perfect Information Action"); 

		plt.yticks([0,1,2,3,4],['Left','Right','Up','Down','Stay'],rotation=90,va='center'); 
		plt.xticks([0,1,2,3,4],['Left','Right','Up','Down','Stay']); 
		plt.xlim([-0.5,4.5]); 
		plt.ylim([-0.5,4.5]);
		plt.title('Probability of Confusion for {}'.format(names[r]));
		#plt.show(); 
		plt.savefig('../../../../../mnt/c/Users/clbur/OneDrive/Work Docs/Presentations/11_5_18 Nisar Meeting/'+names[r]+'Confusion.png'); 


def MMSBasics(save=False):

	data = np.load("../results/MMS/fullMMSResults.npy").item();  
	# data['VB'] = np.load("../results/D4DiffsSoftmax_Data_MMS_300.npy").item(); 
	# data['GM'] = np.load("../results/D4Diffs_Data_MMS_300.npy").item(); 
	# data['Greedy'] = np.load("../results/D4Diffs_Data_MMS_Greedy_300.npy").item(); 


	totalNum = 1000; 

	allKeys = ['VB','GM','Greedy']; 
	catches = {'VB':[],'GM':[],'Greedy':[]}; 
	for key in data.keys():
		#for r in data[key]['Rewards']:
		for r in data[key]['Rewards'][-totalNum:]:
			if(5 in r):
				catches[key].append(r.index(5)); 


	for key in allKeys:
		print("For Run: {}".format(key))
		print("Percent Caught: {:0.2f}".format(len(catches[key])*100/(totalNum))); 
		print("Mean Catch Given Caught: {:0.2f}".format(np.mean(catches[key]))); 
		print("")

		# plt.hist(catches[key],bins=50); 
		# plt.show(); 


	allTests = np.zeros(shape=(3,3)); 
	cellColors = np.zeros(shape=(3,3,3)); 
	for key in allKeys:
		for key2 in allKeys:

			z,p = sm.stats.proportions_ztest([len(catches[key]),len(catches[key2])],[totalNum,totalNum]); 
			allTests[allKeys.index(key),allKeys.index(key2)] = "{:0.5f}".format(p)
			#cellColors[allKeys.index(key),allKeys.index(key2)] = (1-p,p,0); 
			if(p> 0.1):
				cellColors[allKeys.index(key),allKeys.index(key2)] = (1,0,0); 
			elif(p>0.05):
				cellColors[allKeys.index(key),allKeys.index(key2)] = (0.5,0.5,0); 
			else:
				cellColors[allKeys.index(key),allKeys.index(key2)] = (0,1,0); 



	fig,axarr =plt.subplots(2,1); 
	ct=[["{:0.2f}".format(len(catches['VB'])*100/(totalNum)),"{:0.2f}".format(len(catches['GM'])*100/(totalNum)),"{:0.2f}".format(len(catches['Greedy'])*100/(totalNum))],["{:0.2f}".format(np.mean(catches['VB'])),"{:0.2f}".format(np.mean(catches['GM'])),"{:0.2f}".format(np.mean(catches['Greedy']))]]
	axarr[0].table(cellText=ct,colLabels=('VB','GM','Greedy'),rowLabels=('%Caught','Mean Step'),loc='center'); 
	axarr[1].table(cellText=allTests,loc='center',colLabels=('VB','GM','Greedy'), rowLabels=('VB','GM','Greedy'),cellColours=cellColors);
	axarr[0].set_title("MMS Test Stats")
	axarr[1].set_title("Binomial Test P-Values")

	axarr[0].axis('tight'); 
	axarr[0].axis('off'); 
	axarr[0].grid('off'); 
	axarr[1].axis('tight');  
	axarr[1].axis('off'); 
	axarr[1].grid('off'); 
	if(save):
		plt.savefig('../../../../../mnt/c/Users/clbur/OneDrive/Work Docs/Presentations/11_5_18 Nisar Meeting/MMSBasic.png'); 
	else:
		plt.show(); 
	

def MMSActions(save = False):
	data = np.load("../results/MMS/fullMMSResults.npy").item();  
	# data['VB'] = np.load("../results/D4DiffsSoftmax_Data_MMS_300.npy").item(); 
	# data['GM'] = np.load("../results/D4Diffs_Data_MMS_300.npy").item(); 
	# data['Greedy'] = np.load("../results/D4Diffs_Data_MMS_Greedy_300.npy").item(); 

	runs = [data['VB'],data['GM'],data['Greedy']]; 
	names = ["VB","GM","Greedy"];

	for r in range(0,len(runs)):
		plt.cla(); 
		plt.clf(); 

		print(names[r]); 

		acts = runs[r]['Actions']; 
		states = runs[r]['States']; 
		rewards = runs[r]['Rewards'];

		colors = ['b','r','g','m','y']; 
		allColors = []; 
		for i in range(0,len(acts)): 
			for j in range(1,len(acts[i][0])-1):
				allColors.append(colors[acts[i][0][j]]); 


	#	print(np.array(acts).shape)

		allX = []; 
		allY = []; 
		for i in range(0,len(states)):
			for j in range(1,len(states[i][0])-1):
				#print(states[i][0][j][0])
			
				allX.append(states[i][0][j][0]); 
				allY.append(states[i][0][j][1]); 


		# allLines = []; 
		# for i in range(0,len(states)):
		# 	if(5 not in runs[r]['Rewards'][i]):
		# 		continue; 
		# 	allLines.append([[],[]]); 
		# 	for j in range(1,len(states[i][0])-1):
		# 		#print(states[i][0][j][0])
		# 		allLines[-1][0].append([states[i][0][j][0]]);
		# 		allLines[-1][1].append([states[i][0][j][1]]); 


		# print(len(allX)); 
		# print(len(allY));

		plt.axes().set_aspect('equal');
		plt.scatter(allX,allY,color=allColors,s=1);
		
		plt.xlabel("$\Delta$X") 
		plt.ylabel("$\Delta$Y") 
		plt.xlim([-10,10]); 
		plt.ylim([-10,10]);
		
		plt.title(names[r] + " Actions"); 
		if(save):
			#plt.savefig('../img/'+names[r]+"MMSActions.png");
			plt.savefig('../../../../../mnt/c/Users/clbur/OneDrive/Work Docs/Presentations/11_5_18 Nisar Meeting/'+names[r]+'MMSActions.png'); 
		else:
			plt.show(); 

def MMSTraces(save = False):
	data = np.load("../results/MMS/fullMMSResults.npy").item();   
	# data['VB'] = np.load("../results/D4DiffsSoftmax_Data_MMS_300.npy").item(); 
	# data['GM'] = np.load("../results/D4Diffs_Data_MMS_300.npy").item(); 
	# data['Greedy'] = np.load("../results/D4Diffs_Data_MMS_Greedy_300.npy").item(); 

	runs = [data['VB'],data['GM'],data['Greedy']]; 
	names = ["VB","GM","Greedy"];

	for r in range(0,len(runs)):
		plt.cla(); 
		plt.clf(); 

		print(names[r]); 

		acts = runs[r]['Actions']; 
		states = runs[r]['States']; 

		colors = ['b','r','g','m','y']; 
		allColors = []; 
		for i in range(0,len(acts)): 
			for j in range(1,len(acts[i][0])-1):
				allColors.append(colors[acts[i][0][j]]); 


	#	print(np.array(acts).shape)

		# allX = []; 
		# allY = []; 
		# for i in range(0,len(states)):
		# 	for j in range(1,len(states[i][0])-1):
		# 		#print(states[i][0][j][0])
		# 		allX.append(states[i][0][j][0]); 
		# 		allY.append(states[i][0][j][1]); 

		allLines = []; 
		for i in range(0,len(states)):
			if(5 not in runs[r]['Rewards'][i]):
				continue; 
			allLines.append([[],[]]); 
			for j in range(1,len(states[i][0])-1):
				#print(states[i][0][j][0])
				allLines[-1][0].append([states[i][0][j][0]]);
				allLines[-1][1].append([states[i][0][j][1]]); 


		# print(len(allX)); 
		# print(len(allY));

		plt.axes().set_aspect('equal');
		#plt.scatter(allX,allY,color=allColors,s=3);
		num = len(allLines);
		#num = 5
		for i in range(0,num):
			for j in range(1,len(allLines[i][0])):
				plt.plot(allLines[i][0][j-1:j+1],allLines[i][1][j-1:j+1],color=(j/len(allLines[i][0]),1-(j/len(allLines[i][0])),0),linewidth=0.5); 
				#plt.plot(allLines[i][0][j-1:j+1],allLines[i][1][j-1:j+1],color=(j/len(allLines[i][0]),0,0));
		plt.xlabel("$\Delta$X") 
		plt.ylabel("$\Delta$Y") 
		plt.xlim([-10,10]); 
		plt.ylim([-10,10]);
		
		plt.title(names[r] + " MMS Traces"); 
		if(save):
			#plt.savefig('../img/'+names[r]+"MMSTraces.png");
			plt.savefig('../../../../../mnt/c/Users/clbur/OneDrive/Work Docs/Presentations/11_5_18 Nisar Meeting/'+names[r]+'MMSTracesAll.png'); 
		else:
			plt.show(); 


def MMSConfusion(save=False):


	data = np.load("../results/MMS/fullMMSResults.npy").item();  
	# data['VB'] = np.load("../results/D4DiffsSoftmax_Data_MMS_300.npy").item(); 
	# data['GM'] = np.load("../results/D4Diffs_Data_MMS_300.npy").item(); 
	# data['Greedy'] = np.load("../results/D4Diffs_Data_MMS_Greedy_300.npy").item(); 

	runs = [data['VB'],data['GM'],data['Greedy']]; 
	names = ["VB","GM","Greedy"];

	for r in range(0,len(runs)):
		plt.cla(); 
		plt.clf(); 

		print(names[r]); 

		acts = runs[r]['Actions']; 
		states = runs[r]['States']; 

		confusion = np.zeros(shape=(5,5)); 

		for i in range(0,len(states)):
			for j in range(1,len(states[i][0])-1):
				# #print(states[i][0][j][0])
				# allX.append(states[i][0][j][0]); 
				# allY.append(states[i][0][j][1]); 
				x = states[i][0][j]; 
				#Determine Quadrant
				
				if(np.sqrt(x[0]**2 + x[1]**2) < 1):
					first = 4; 
				else:
					if(abs(x[0])>abs(x[1])):
						if(x[0] > 0):
							#action should be left
							first = 0; 
						else:
							#action should be right
							first = 1;
					else:
						if(x[1] > 0):
							first = 3; 
						else:
							first = 2; 
				second = acts[i][0][j];
				confusion[first][second] += 1; 

		confusion += 0.00000000001
		for i in range(0,5):
			if(sum(confusion[i])>0):
				confusion[i] = confusion[i]/sum(confusion[i])
		

		plt.imshow(np.log(confusion),vmin=-10,vmax=0);
		plt.colorbar(); 
		plt.xlabel("Observed Action"); 
		plt.ylabel("Perfect Information Action"); 

		plt.yticks([0,1,2,3,4],['Left','Right','Up','Down','Stay'],rotation=90,va='center'); 
		plt.xticks([0,1,2,3,4],['Left','Right','Up','Down','Stay']); 
		plt.xlim([-0.5,4.5]); 
		plt.ylim([-0.5,4.5]);
		plt.title('Log-Probability of Confusion for {}'.format(names[r]));

		if(save):
			plt.savefig('../../../../../mnt/c/Users/clbur/OneDrive/Work Docs/Presentations/11_5_18 Nisar Meeting/'+names[r]+'MMSLogConfusion.png'); 
		else:
			plt.show(); 

		plt.clf(); 
		plt.cla(); 
		plt.imshow(confusion,vmin=0,vmax=1);
		plt.colorbar(); 
		plt.xlabel("Observed Action"); 
		plt.ylabel("Perfect Information Action"); 

		plt.yticks([0,1,2,3,4],['Left','Right','Up','Down','Stay'],rotation=90,va='center'); 
		plt.xticks([0,1,2,3,4],['Left','Right','Up','Down','Stay']); 
		plt.xlim([-0.5,4.5]); 
		plt.ylim([-0.5,4.5]);
		plt.title('Probability of Confusion for {}'.format(names[r]));
		
		if(save):
			plt.savefig('../../../../../mnt/c/Users/clbur/OneDrive/Work Docs/Presentations/11_5_18 Nisar Meeting/'+names[r]+'MMSConfusion.png'); 
		else:
			plt.show(); 


def MMSDensity(save = False):
	data = np.load("../results/MMS/fullMMSResults.npy").item();  
	# data['VB'] = np.load("../results/D4DiffsSoftmax_Data_MMS_300.npy").item(); 
	# data['GM'] = np.load("../results/D4Diffs_Data_MMS_300.npy").item(); 
	# data['Greedy'] = np.load("../results/D4Diffs_Data_MMS_Greedy_300.npy").item(); 

	runs = [data['VB'],data['GM'],data['Greedy']]; 
	names = ["VB","GM","Greedy"];

	for r in range(0,len(runs)):
		plt.cla(); 
		plt.clf(); 

		print(names[r]); 

		acts = runs[r]['Actions']; 
		states = runs[r]['States']; 
		rewards = runs[r]['Rewards']

		colors = ['b','r','g','m','y']; 
		allColors = []; 
		for i in range(0,len(acts)): 
			for j in range(1,len(acts[i][0])-1):
				allColors.append(colors[acts[i][0][j]]); 


	#	print(np.array(acts).shape)

		allX = []; 
		allY = []; 
		for i in range(0,len(states)):
			for j in range(1,len(states[i][0])-1):
			# for j in range(0,1):
				#print(states[i][0][j][0])
				if(5 in rewards[i]):
					allX.append(states[i][0][j][0]); 
					allY.append(states[i][0][j][1]); 

		# allLines = []; 
		# for i in range(0,len(states)):
		# 	if(5 not in runs[r]['Rewards'][i]):
		# 		continue; 
		# 	allLines.append([[],[]]); 
		# 	for j in range(1,len(states[i][0])-1):
		# 		#print(states[i][0][j][0])
		# 		allLines[-1][0].append([states[i][0][j][0]]);
		# 		allLines[-1][1].append([states[i][0][j][1]]); 


		# print(len(allX)); 
		# print(len(allY));
		


		plt.axes().set_aspect('equal');
		#plt.scatter(allX,allY,color=allColors,s=3);
		plt.hist2d(allX,allY,range=[[-9,9],[-9,9]],bins=41,normed=True,vmin=0,vmax=0.015); 
		plt.colorbar(); 

		plt.xlabel("$\Delta$X") 
		plt.ylabel("$\Delta$Y") 
		#plt.xlim([-10,10]); 
		#plt.ylim([-10,10]);
		
		plt.title(names[r] + " MMS Density"); 
		if(save):
			#plt.savefig('../img/'+names[r]+"MMSDensity.png");
			plt.savefig('../../../../../mnt/c/Users/clbur/OneDrive/Work Docs/Presentations/11_5_18 Nisar Meeting/'+names[r]+'MMSDensity.png'); 
		else:
			plt.show(); 


def makeUniObsModel(save=False):

	cent = [0,0]; 
	length = 2; 
	width = 2; 
	orient = 0; 
	steep = 0.05; 

	pz = Softmax(); 
	pz.buildOrientedRecModel(cent,orient,length,width,steepness=steep); 

	[x,y,dom] = pz.plot2D(low=[-10,-10],high=[10,10],vis=False);

	plt.figure(dpi=200); 
	plt.contourf(x,y,dom); 
	plt.text(0,0,'Detect',color='r',fontsize=10,horizontalalignment='center',verticalalignment='center',fontweight='bold'); 
	plt.text(0,5,'North',color='r',fontsize=15,horizontalalignment='center',verticalalignment='center',fontweight='bold'); 
	plt.text(5,0,'West',color='r',fontsize=15,horizontalalignment='center',verticalalignment='center',fontweight='bold'); 
	plt.text(0,-5,'South',color='r',fontsize=15,horizontalalignment='center',verticalalignment='center',fontweight='bold'); 
	plt.text(-5,0,'East',color='r',fontsize=15,horizontalalignment='center',verticalalignment='center',fontweight='bold'); 

	plt.plot([-10,-1],[-10,-1],color='red',linestyle='--',linewidth=3)
	plt.plot([10,1],[-10,-1],color='red',linestyle='--',linewidth=3)
	plt.plot([10,1],[10,1],color='red',linestyle='--',linewidth=3)
	plt.plot([-10,-1],[10,1],color='red',linestyle='--',linewidth=3)
	plt.xlim([-9.9,9.9]); 
	plt.ylim([-9.9,9.9]); 
	plt.xlabel("$\Delta$X") 
	plt.ylabel("$\Delta$Y") 

	plt.title("2D Search Observation Model",fontsize=20)
	plt.axes().set_aspect('equal');

	if(save):
		plt.savefig('../../../../../mnt/c/Users/clbur/OneDrive/Work Docs/Journals/TRO 2018/postRev/UniObsModel.pdf');
	else:
		plt.show(); 

def makeMMSObsModel(save=False):

	cent = [0,0]; 
	length = 2; 
	width = 2; 
	orient = 0; 
	steep = 0.05; 

	pz = Softmax(); 
	pz.buildOrientedRecModel(cent,orient,length,width,steepness=steep); 

	[x,y,dom] = pz.plot2D(low=[-10,-10],high=[10,10],vis=False);

	for i in range(0,len(dom)):
		for j in range(0,len(dom[i])):
			if(dom[i][j] == 0):
				dom[i][j] = 1; 
			else:
				dom[i][j]=0; 

	plt.figure();  
	plt.contourf(x,y,dom,cmap='inferno'); 
	plt.text(0,0,'Detect',color='r',fontsize=8,horizontalalignment='center',verticalalignment='center',fontweight='bold'); 
	plt.text(0,5,'No Detect',color='r',fontsize=10,horizontalalignment='center',verticalalignment='center',fontweight='bold'); 
	plt.text(5,0,'No Detect',color='r',fontsize=10,horizontalalignment='center',verticalalignment='center',fontweight='bold'); 
	plt.text(0,-5,'No Detect',color='r',fontsize=10,horizontalalignment='center',verticalalignment='center',fontweight='bold'); 
	plt.text(-5,0,'No Detect',color='r',fontsize=10,horizontalalignment='center',verticalalignment='center',fontweight='bold'); 

	plt.plot([-10,-1],[-10,-1],color='red',linestyle='--',linewidth=3)
	plt.plot([10,1],[-10,-1],color='red',linestyle='--',linewidth=3)
	plt.plot([10,1],[10,1],color='red',linestyle='--',linewidth=3)
	plt.plot([-10,-1],[10,1],color='red',linestyle='--',linewidth=3)
	plt.xlim([-9.9,9.9]); 
	plt.ylim([-9.9,9.9]); 
	plt.xlabel("$\Delta$X") 
	plt.ylabel("$\Delta$Y") 

	plt.title("Multi-Modal Softmax Observation Model")
	plt.axes().set_aspect('equal');

	if(save):
		plt.savefig('../../../../../mnt/c/Users/clbur/OneDrive/Work Docs/Presentations/11_5_18 Nisar Meeting/ObsModelMMS.png');
	else:
		plt.show(); 


def combineMMSData():
	data = {}; 
	data['VB1'] = np.load("../results/MMS/D4DiffsSoftmax_Data_MMS_500.npy").item(); 
	data['GM1'] = np.load("../results/MMS/D4Diffs_Data_MMS_500.npy").item(); 
	data['Greedy1'] = np.load("../results/MMS/D4Diffs_Data_MMS_Greedy_400.npy").item(); 

	data['VB2'] = np.load("../results/MMS/D4DiffsSoftmax_Data_MMS_250.npy").item(); 
	data['GM2'] = np.load("../results/MMS/D4Diffs_Data_MMS_250.npy").item(); 
	data['Greedy2'] = np.load("../results/MMS/D4Diffs_Data_MMS_Greedy_300.npy").item(); 

	data['VB3'] = np.load("../results/MMS/D4DiffsSoftmax_Data_MMS_Sec250.npy").item(); 
	data['GM3'] = np.load("../results/MMS/D4Diffs_Data_MMS_Sec250.npy").item(); 
	data['Greedy3'] = np.load("../results/MMS/D4Diffs_Data_MMS_Sec_Greedy_300.npy").item(); 

	VBData = {'States':[], 'Actions':[],'Rewards':[]}; 
	k = ['VB1','VB2','VB3']; 
	for key in k: 
		for key2 in data[key].keys():
			for a in data[key][key2]:
				VBData[key2].append(a); 

	GMData = {'States':[], 'Actions':[],'Rewards':[]}; 
	k = ['GM1','GM2','GM3']; 
	for key in k: 
		for key2 in data[key].keys():
			for a in data[key][key2]:
				GMData[key2].append(a); 

	GreedyData = {'States':[], 'Actions':[],'Rewards':[]}; 
	k = ['Greedy1','Greedy2','Greedy3']; 
	for key in k: 
		for key2 in data[key].keys():
			for a in data[key][key2]:
				GreedyData[key2].append(a); 

	data = {'VB':VBData,'GM':GMData,'Greedy':GreedyData}; 

	np.save('../results/MMS/fullMMSResults.npy',data); 


def makeStoryboard(save=False):

	data = np.load('../results/MMS/StoryBoardMedium.npy').item()
	
	data['States'] = data['States'][0][0];
	
	stateData = [[],[]]; 
	for i in range(0,len(data['States'])):
		stateData[0].append(data['States'][i][0]); 
		stateData[1].append(data['States'][i][1]);  


	ims = []; 
	for i in range(0,len(data['Beliefs'])):
		plt.cla(); 
		plt.clf(); 
		b = data['Beliefs'][i].slice2DFrom4D(retGS=True,dims=[0,1],vis=False);
		[x,y,c] = b.plot2D(low=[-10,-10],high=[10,10],vis=False); 

		colors = []; 
		for j in range(0,i):
			colors.append([1,0,0,(j/i)**2]); 
		plt.contourf(x,y,c); 

		for j in range(0,i):
			#plt.plot(allLines[i][0][j-1:j+1],allLines[i][1][j-1:j+1],color=(j/len(allLines[i][0]),1-(j/len(allLines[i][0])),0),linewidth=0.5); 
			plt.plot(stateData[0][j-1:j+1],stateData[1][j-1:j+1],color=colors[j],linewidth=3); 
		plt.scatter(stateData[0][i-1],stateData[1][i-1],c='r',s=35)
		#plt.scatter(stateData[0][0:i],stateData[1][0:i],c=colors,s=20,marker='x'); 


		plt.xlim([-10,10]); 
		plt.ylim([-10,10])
		plt.pause(0.1)



	# plt.cla(); 
	# plt.clf(); 

	# acts = data['Actions']; 
	# states = data['States']; 
	# rewards = data['Rewards'];

	# colors = ['b','r','g','m','y']; 
	# allColors = []; 
	# for i in range(0,len(acts)): 
	# 	for j in range(1,len(acts[i][0])-1):
	# 		allColors.append(colors[acts[i][0][j]]); 




			
				#plt.plot(allLines[i][0][j-1:j+1],allLines[i][1][j-1:j+1],color=(j/len(allLines[i][0]),0,0));


	# allX = []; 
	# allY = []; 
	# for i in range(0,len(states)):
	# 	for j in range(1,len(states[i][0])-1):
	# 		#print(states[i][0][j][0])
		
	# 		allX.append(states[i][0][j][0]); 
	# 		allY.append(states[i][0][j][1]); 

	# plt.axes().set_aspect('equal');
	# plt.scatter(allX,allY,color=allColors,s=1);
	
	# plt.xlabel("$\Delta$X") 
	# plt.ylabel("$\Delta$Y") 
	# plt.xlim([-10,10]); 
	# plt.ylim([-10,10]);
	
	# plt.title("Short Actions"); 
	# if(save):
	# 	plt.savefig('../../../../../mnt/c/Users/clbur/OneDrive/Work Docs/Presentations/11_5_18 Nisar Meeting/'+names[r]+'MMSActions.png'); 
	# else:
	# 	plt.show(); 


def POMCPExploration(save=False):

	folder = '../../../../../mnt/d/POMCP/'; 
	#files = ['fifthSecondPOMCP_2_c10.txt', 'halfSecondPOMCP_2_c10.txt', 'oneSecondPOMCP_2_c10.txt', 'tenthSecondPOMCP_2_c10.txt', 'thirdSecondPOMCP_2_c10.txt', 'threeSecondPOMCP_2_c10.txt', 'twentithSecondPOMCP_2_c10.txt', 'twoSecondPOMCP_2_c10.txt']
	#times = [.2,.5,1,.1,1/3,3,0.05,2]; 
	files = ['twentithSecondPOMCP_2_c10.txt','tenthSecondPOMCP_2_c10.txt','fifthSecondPOMCP_2_c10.txt','thirdSecondPOMCP_2_c10.txt','halfSecondPOMCP_2_c10.txt','threeForthSecondPOMCP_2_c10.txt','oneSecondPOMCP_2_c10.txt','twoSecondPOMCP_2_c10.txt','threeSecondPOMCP_2_c10.txt']
	times = [0.05,.1,.2,1/3,.5,.75,1,2,3];

	allMeans = []; 
	threeTotals = []; 

	for i in range(0,len(files)):
		fullName = folder+files[i]; 
		file = open(fullName,'r'); 

		data = file.readlines(); 

		allTotals = []; 

		for k in range(0,len(data)):
			tmp = data[k].split('='); 
			total = 0; 
			for j in range(0,len(tmp)):
				if('sp' in tmp[j]):
					total += float(tmp[j][1:4]); 
			allTotals.append(total); 

		#print(allTotals); 
		# print(files[i]); 
		# print("Mean: {}".format(np.mean(allTotals))); 
		# print("SD: {}".format(np.std(allTotals))); 
		# print(""); 
		allMeans.append(np.mean(allTotals)); 
		if(i == len(files)-1):
			threeTotals = allTotals; 


	plt.plot(times,allMeans,c='m',linewidth=2); 
	plt.axhline(110.79,c='g',linestyle='--',linewidth=2);
	plt.axhline(101.17,c='b',linestyle='--',linewidth=2); 

	for t in times:
		plt.axvline(t,c='m',linestyle='--',linewidth=0.5,alpha=0.5)


	plt.ylabel('Average Accumlated Reward'); 
	plt.xlabel('POMCP Decision Time'); 
	plt.legend(["POMCP","VB-POMDP","GM-POMDP"])

	if(save):
		plt.savefig('../../../../../mnt/c/Users/clbur/OneDrive/Work Docs/Presentations/11_5_18 Nisar Meeting/POMCPApproach.png');
	else:
		plt.show(); 

	data = np.load('../results/accumlatedResults.npy').item(); 


	VB = data['VBFinalRewards']; 
	GM = data['GMFinalRewards']; 
	Greedy = data['greedFinalRewards']; 
	Perfect = data['perfectFinalRewards']; 
	

	bins = [i*5+30 for i in range(0,64)]
	

	fig,axarr = plt.subplots(5,1,sharex=True,sharey=True); 
	axarr[1].hist(VB,density=True,bins=bins,color='g'); 
	axarr[1].set_xlabel('VB'); 
	mu,std = norm.fit(VB); 
	xmin,xmax = axarr[1].get_xlim(); 
	x = np.linspace(xmin,xmax,100); 
	p = norm.pdf(x,mu,std); 
	axarr[1].plot(x,p,'k',linewidth=2); 
	axarr[1].set_xlim([0,250])
	axarr[1].set_ylim([0,0.02]); 

	axarr[0].hist(Perfect,density=True,bins=bins,color='k'); 
	axarr[0].set_xlabel('Perfect Info');
	mu,std = norm.fit(Perfect); 
	xmin,xmax = axarr[0].get_xlim(); 
	x = np.linspace(xmin,xmax,100); 
	p = norm.pdf(x,mu,std); 
	axarr[0].plot(x,p,'r',linewidth=2); 
	axarr[0].set_xlim([0,250])
	axarr[0].set_ylim([0,0.02]); 

	axarr[3].hist(GM,density=True,bins=bins,color='b'); 
	axarr[3].set_xlabel('GM');
	mu,std = norm.fit(GM); 
	xmin,xmax = axarr[3].get_xlim(); 
	x = np.linspace(xmin,xmax,100); 
	p = norm.pdf(x,mu,std); 
	axarr[3].plot(x,p,'k',linewidth=2); 
	axarr[3].set_xlim([0,250])
	axarr[3].set_ylim([0,0.02]); 

	axarr[4].hist(Greedy,density=True,bins=bins,color='r'); 
	axarr[4].set_xlabel('Greedy');
	mu,std = norm.fit(Greedy); 
	xmin,xmax = axarr[4].get_xlim(); 
	x = np.linspace(xmin,xmax,100); 
	p = norm.pdf(x,mu,std); 
	axarr[4].plot(x,p,'k',linewidth=2); 
	axarr[4].set_xlim([0,250])
	axarr[4].set_ylim([0,0.02]); 

	axarr[2].hist(threeTotals,density=True,bins=bins,color='m'); 
	mu,std = norm.fit(threeTotals); 
	xmin,xmax = axarr[2].get_xlim(); 
	x = np.linspace(xmin,xmax,100); 
	p = norm.pdf(x,mu,std); 
	axarr[2].plot(x,p,'k',linewidth=2); 
	axarr[2].set_xlim([0,250])
	axarr[2].set_ylim([0,0.02]); 
	axarr[2].axvline(np.mean(threeTotals),c='k'); 
	axarr[2].set_xlabel('3s POMCP');


	axarr[1].axvline(np.mean(VB),c='k'); 
	axarr[0].axvline(np.mean(Perfect),c='r'); 
	axarr[3].axvline(np.mean(GM),c='k'); 
	axarr[4].axvline(np.mean(Greedy),c='k'); 
	axarr[2].axvline(np.mean(threeTotals),c='k'); 
	plt.subplots_adjust(hspace=0.35)
	fig.suptitle("HighQ Histograms")
	#fig.text(0.04,0.5,'PDF', va = 'center',rotation='vertical')
	axarr[2].set_ylabel('PDF')

	if(save):
		plt.savefig('../../../../../mnt/c/Users/clbur/OneDrive/Work Docs/Presentations/11_5_18 Nisar Meeting/POMCPHistComp.png');
	else:
		plt.show(); 


def remake1DBanner(save = False):

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

	my_dpi = 200; 
	fig = plt.figure(figsize=(1200/my_dpi,1200/my_dpi),dpi=my_dpi); 
	ax = fig.gca(projection='3d');
	a = ax.plot_surface(xdetectGM,ydetectGM,cdetectGM,color='g',label='Detect');
	b = ax.plot_surface(xNoDetectGM,yNoDetectGM,cNoDetectGM,color='r',label='No Detect');
	ax.view_init(elev=26,azim=-125); 
	ax.set_xlabel("Cop Position (m)",fontsize=20)
	ax.set_ylabel("Robber Position (m)",fontsize=20)
	ax.zaxis.set_rotate_label(False); 
	ax.set_zlabel("$p(o|s)$",rotation=90,fontsize=20)
	ax.set_aspect('equal');
	ax.set_title("GM Likelihood Model",fontsize=30)
	#a.set_edgecolors(('y','k')); 
	#b.set_edgecolors(('y','k')); 
	a._edgecolors2d = ('g',); 
	b._edgecolors2d = ('r',);
	a._facecolors2d = ('g',); 
	b._facecolors2d = ('r',);
	#print(a.get_edgecolors()); 
	#print(b.get_edgecolors()); 

	ax.legend(loc='center right',fontsize=12);
	#plt.tight_layout();
	if(save):
		plt.savefig('../../../../../mnt/c/Users/clbur/OneDrive/Work Docs/Journals/TRO 2018/postRev/ColinearBannerGM.pdf');
	else:
		plt.show();

	#levels = [i/100 for i in range(0,105)];  

	print(np.amax(cNoDetectGM)); 
	print(np.mean(np.array(cNoDetectGM) + np.array(cdetectGM))); 

	#4
	steepness = 5; 
	weight = [[-1.3926,1.3926],[-0.6963,0.6963],[0,0]];
	bias = [0,.4741,0]; 
	weight = (np.array(weight)*steepness).tolist(); 
	bias = (np.array(bias)*steepness).tolist(); 

	low = [0,0]; 
	high = [5,5]; 

	likelihood = Softmax(weight,bias);
	delta = 0.1; 

	x, y = np.mgrid[low[0]:high[0]:delta, low[1]:high[1]:delta]
	pos = np.dstack((x, y))  
	resx = int((high[0]-low[0])//delta)+1;
	resy = int((high[1]-low[1])//delta)+1; 

	model = [[[0 for i in range(0,resy)] for j in range(0,resx)] for k in range(0,len(likelihood.weights))];
	

	for m in range(0,len(likelihood.weights)):
		for i in range(0,resx):
			xx = (i*(high[0]-low[0])/resx + low[0]);
			for j in range(0,resy):
				yy = (j*(high[1]-low[1])/resy + low[1])
				dem = 0; 
				for k in range(0,len(likelihood.weights)):
					dem+=np.exp(likelihood.weights[k][0]*xx + likelihood.weights[k][1]*yy + likelihood.bias[k]);
				model[m][i][j] = np.exp(likelihood.weights[m][0]*xx + likelihood.weights[m][1]*yy + likelihood.bias[m])/dem;

	for i in range(0,len(model[0])):
		for j in range(0,len(model[0][i])):
			model[0][i][j] += model[2][i][j]; 
	softDetect = np.array(model[1]); 
	softNoDetect = np.array(model[0]); 


	fig = plt.figure(figsize=(1200/my_dpi,1200/my_dpi),dpi=my_dpi); 
	ax = fig.gca(projection='3d');
	a = ax.plot_surface(x,y,softDetect,color='g',label='Detect');
	b = ax.plot_surface(x,y,softNoDetect,color='r',label='No Detect');
	ax.view_init(elev=26,azim=-125); 
	ax.set_xlabel("Cop Position (m)",fontsize=20)
	ax.set_ylabel("Robber Position (m)",fontsize=20)
	ax.zaxis.set_rotate_label(False); 
	ax.set_zlabel("$p(o|s)$",rotation=90,fontsize=20)
	ax.set_aspect('equal');
	ax.set_title("Softmax Likelihood Model",fontsize=30)
	a._edgecolors2d = ('g',); 
	b._edgecolors2d = ('r',);
	a._facecolors2d = ('g',); 
	b._facecolors2d = ('r',);
	#print(a.get_edgecolors()); 
	#print(b.get_edgecolors()); 

	ax.legend(loc='center right',fontsize=12);
	#plt.tight_layout();
	if(save):
		plt.savefig('../../../../../mnt/c/Users/clbur/OneDrive/Work Docs/Journals/TRO 2018/postRev/ColinearBannerVB.pdf');
	else:
		plt.show();


def remakeAhmedTROSoftamx(save):

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
	#plt.rc('text', usetex=True)
	my_dpi = 200; 
	fig = plt.figure(dpi=my_dpi); 
	ax = fig.gca();
	lw = 6; 
	ax.plot(xprior,cprior,color='g',linewidth=lw); 
	ax.plot(xlikelihood,classes[softClass],color='b',linewidth=lw);
	ax.plot(xvlb,cvlb,color='k',linewidth=lw,linestyle='dashed'); 
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
	
	if(save):
		plt.savefig('../../../../../mnt/c/Users/clbur/OneDrive/Work Docs/Journals/TRO 2018/postRev/VB_Fusion_1D_Newest.pdf');
	else:
		plt.show();


def remake2DFusionPlot(save):
	
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
	


	if(save):
		plt.savefig('../../../../../mnt/c/Users/clbur/OneDrive/Work Docs/Journals/TRO 2018/postRev/VBFusion2D.pdf');
	else:
		plt.show();



def remakeCondensationPlot(save):
	
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

	fig,axarr = plt.subplots(1,3,sharey = True,dpi=200); 
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
	fig.suptitle("Condensation of {} mixands to {}".format(start,mid*final),y=0.78,fontsize=15); 
	#plt.savefig('../img/CondensationRemake.png',figsize = (9,3)); 
	plt.tight_layout(); 
	#plt.margins(0,0); 
	if(save):
		plt.savefig('../../../../../mnt/c/Users/clbur/OneDrive/Work Docs/Journals/TRO 2018/postRev/CondensationRemake.pdf',bbox_inches='tight',pad_inches=0,figsize=(9,3));
	else:
		plt.show();


if __name__ == '__main__':

	rc('text', usetex=True)

	#plotCrossStich(); 
	#extractAndSaveAllData(); 



	#makeLayeredPlot(); 

	#makeHighQPlots(False);
	#makeLowQPlots(); 

	#SDExploration(); 

	#actionInvestigation(); 

	#actionConfusion(); 


	#MMSBasics(save=False); 
	#MMSActions(True); 
	#MMSTraces(True); 
	#MMSDensity(True);

	#MMSConfusion(save=False); 

	makeUniObsModel(True)
	#makeMMSObsModel(); 
	
	#makeStoryboard(False);

	#combineMMSData();


	#POMCPExploration(False); 

	#remake1DBanner(True);

	#remakeAhmedTROSoftamx(True); 

	#remake2DFusionPlot(True);

	#remakeCondensationPlot(True); 