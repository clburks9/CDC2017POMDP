import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


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


def make8HourPlot(show = False):

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


	VBFinal = np.load("../results/TRO_Results_Final/D4DiffsSoftmax/D4DiffsSoftmax_Data4.npy",encoding='latin1').item(); 
	GMFinal = np.load("../results/TRO_Results_Final/D4Diffs/D4Diffs_Data5.npy",encoding='latin1').item(); 
	GreedyFinal = np.load("../results/TRO_Results_Final/D4Diffs/D4Diffs_Data_Greedy4.npy",encoding='latin1').item(); 

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

	# print(np.std(VBFinalRewards))
	plt.axhline(np.mean(VBFinalRewards),c='g',linestyle='--');
	plt.axhline(np.mean(GMFinalRewards),c='b',linestyle='--'); 
	plt.axhline(np.mean(greedFinalRewards),c='r',linestyle='--')
	# plt.errorbar(0,np.mean(VBFinalRewards),yerr=np.std(VBFinalRewards));  
	plt.title("Comparison of Policies at various solution times")
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
	GreedyFinal = np.load("../results/TRO_Results_Final/D4Diffs/D4Diffs_Data_Greedy4.npy",encoding='latin1').item(); 
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
	GreedyLowQFinal = np.load("../results/D4Diffs_Data_LowQ_Greedy.npy",encoding='latin1').item(); 
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
	#ax2.set_xlim([.25,3.5]);

	plt.show(); 

def makeBoxAndWhisker():
	data = np.load('../results/accumlatedResults.npy').item(); 
	
	VBFinalRewards = data['VBFinalRewards']; 
	GMFinalRewards = data['GMFinalRewards'];
	greedFinalRewards = data['greedFinalRewards']; 
	perfectFinalRewards = data['perfectFinalRewards']; 


	#plt.boxplot([perfectFinalRewards,VBFinalRewards,GMFinalRewards,greedFinalRewards]); 
	plt.boxplot([perfectFinalRewards,VBFinalRewards,GMFinalRewards,greedFinalRewards],whis=1000,labels=['Perfect Info','VB','GM','Greedy']); 
	#plt.boxplot(GMFinalRewards); 
	plt.show(); 


def makeHists():
	data = np.load('../results/accumlatedResults.npy').item(); 
	
	VBFinalRewards = data['VBFinalRewards']; 
	GMFinalRewards = data['GMFinalRewards'];
	greedFinalRewards = data['greedFinalRewards']; 
	perfectFinalRewards = data['perfectFinalRewards']; 


	print("Perfect, Mean: {}, STD: {}".format(np.mean(perfectFinalRewards),np.std(perfectFinalRewards)))
	print("VB, Mean: {}, STD: {}".format(np.mean(VBFinalRewards),np.std(VBFinalRewards))); 
	print("GM, Mean: {}, STD: {}".format(np.mean(GMFinalRewards),np.std(GMFinalRewards))); 
	print("Greedy, Mean: {}, STD: {}".format(np.mean(greedFinalRewards),np.std(greedFinalRewards))); 


	fig,axarr = plt.subplots(2,2,sharex=True,sharey=True); 
	axarr[0][0].hist(VBFinalRewards,density=True); 
	axarr[0][0].set_xlabel('VB'); 
	axarr[0][1].hist(perfectFinalRewards,density=True); 
	axarr[0][1].set_xlabel('Perfect Info');
	axarr[1][0].hist(GMFinalRewards,density=True); 
	axarr[1][0].set_xlabel('GM');
	axarr[1][1].hist(greedFinalRewards,density=True); 
	axarr[1][1].set_xlabel('Greedy');

	axarr[0][0].axvline(np.mean(VBFinalRewards),c='k'); 
	axarr[0][1].axvline(np.mean(perfectFinalRewards),c='k'); 
	axarr[1][0].axvline(np.mean(GMFinalRewards),c='k'); 
	axarr[1][1].axvline(np.mean(greedFinalRewards),c='k'); 

	plt.show(); 



def makeHighQPlots():

	data = np.load('../results/accumlatedResults.npy').item(); 


	VB = data['VBFinalRewards']; 
	GM = data['GMFinalRewards']; 
	Greedy = data['greedFinalRewards']; 
	Perfect = data['perfectFinalRewards']; 

	print("Perfect, Mean: {}, STD: {}".format(np.mean(Perfect),np.std(Perfect)))
	print("VB, Mean: {}, STD: {}".format(np.mean(VB),np.std(VB))); 
	print("GM, Mean: {}, STD: {}".format(np.mean(GM),np.std(GM))); 
	print("Greedy, Mean: {}, STD: {}".format(np.mean(Greedy),np.std(Greedy))); 

	plt.figure();
	plt.boxplot([Perfect,VB,GM,Greedy],whis=1000,labels=['Perfect Info','VB','GM','Greedy']); 
	plt.title('High Q 1000-Sim Rewards Distributions'); 
	#plt.set_aspect('equal'); 
	plt.savefig('../../../../../mnt/c/Users/clbur/OneDrive/Work Docs/Presentations/11_5_18 Nisar Meeting/HighQBox.png'); 
	#plt.boxplot(GMFinalRewards); 
	#plt.show(); 

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

	axarr[0].hist(Perfect,density=True,bins=bins,color='k'); 
	axarr[0].set_xlabel('Perfect Info');
	mu,std = norm.fit(Perfect); 
	xmin,xmax = axarr[0].get_xlim(); 
	x = np.linspace(xmin,xmax,100); 
	p = norm.pdf(x,mu,std); 
	axarr[0].plot(x,p,'r',linewidth=2); 
	axarr[0].set_xlim([0,250])

	axarr[2].hist(GM,density=True,bins=bins,color='b'); 
	axarr[2].set_xlabel('GM');
	mu,std = norm.fit(GM); 
	xmin,xmax = axarr[2].get_xlim(); 
	x = np.linspace(xmin,xmax,100); 
	p = norm.pdf(x,mu,std); 
	axarr[2].plot(x,p,'k',linewidth=2); 
	axarr[2].set_xlim([0,250])

	axarr[3].hist(Greedy,density=True,bins=bins,color='r'); 
	axarr[3].set_xlabel('Greedy');
	mu,std = norm.fit(Greedy); 
	xmin,xmax = axarr[3].get_xlim(); 
	x = np.linspace(xmin,xmax,100); 
	p = norm.pdf(x,mu,std); 
	axarr[3].plot(x,p,'k',linewidth=2); 
	axarr[3].set_xlim([0,250])

	axarr[1].axvline(np.mean(VB),c='k'); 
	axarr[0].axvline(np.mean(Perfect),c='r'); 
	axarr[2].axvline(np.mean(GM),c='k'); 
	axarr[3].axvline(np.mean(Greedy),c='k'); 

	fig.suptitle("HighQ 1000-Sim Histograms")
	#plt.set_aspect('equal');
	plt.savefig('../../../../../mnt/c/Users/clbur/OneDrive/Work Docs/Presentations/11_5_18 Nisar Meeting/HighQHists.png'); 

	#plt.show(); 

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

	plt.figure(); 
	
	plt.boxplot([PerfectLowQ,VBLowQ,GMLowQ,GreedyLowQ],whis=1000,labels=['Perfect Info','VB','GM','Greedy']); 
	plt.title('LowQ 1000-Sim Rewards Distributions'); 
	if(save):
		plt.savefig('../../../../../mnt/c/Users/clbur/OneDrive/Work Docs/Presentations/11_5_18 Nisar Meeting/LowQBox.png'); 
	#plt.boxplot(GMFinalRewards); 
	#plt.show(); 

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

	axarr[0].hist(PerfectLowQ,density=True,bins=bins,color='k'); 
	axarr[0].set_xlabel('Perfect Info');
	mu,std = norm.fit(PerfectLowQ); 
	xmin,xmax = axarr[0].get_xlim(); 
	x = np.linspace(xmin,xmax,100); 
	p = norm.pdf(x,mu,std); 
	axarr[0].plot(x,p,'r',linewidth=2); 
	axarr[0].set_xlim([0,250])

	axarr[2].hist(GMLowQ,density=True,bins=bins,color='b'); 
	axarr[2].set_xlabel('GM');
	mu,std = norm.fit(GMLowQ); 
	xmin,xmax = axarr[2].get_xlim(); 
	x = np.linspace(xmin,xmax,100); 
	p = norm.pdf(x,mu,std); 
	axarr[2].plot(x,p,'k',linewidth=2); 
	axarr[2].set_xlim([0,250])

	axarr[3].hist(GreedyLowQ,density=True,bins=bins,color='r'); 
	axarr[3].set_xlabel('Greedy');
	mu,std = norm.fit(GreedyLowQ); 
	xmin,xmax = axarr[3].get_xlim(); 
	x = np.linspace(xmin,xmax,100); 
	p = norm.pdf(x,mu,std); 
	axarr[3].plot(x,p,'k',linewidth=2); 
	axarr[3].set_xlim([0,250])

	axarr[1].axvline(np.mean(VBLowQ),c='k'); 
	axarr[0].axvline(np.mean(PerfectLowQ),c='r'); 
	axarr[2].axvline(np.mean(GMLowQ),c='k'); 
	axarr[3].axvline(np.mean(GreedyLowQ),c='k');

	fig.suptitle("LowQ 1000-Sim Histograms")
	if(save):
		plt.savefig('../../../../../mnt/c/Users/clbur/OneDrive/Work Docs/Presentations/11_5_18 Nisar Meeting/LowQHists.png'); 

	#plt.show(); 


def SDExploration():

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

	plt.savefig('../../../../../mnt/c/Users/clbur/OneDrive/Work Docs/Presentations/11_5_18 Nisar Meeting/SDExploration.png'); 



def actionInvestigation():

	VBFinal = np.load("../results/TRO_Results_Final/D4DiffsSoftmax/D4DiffsSoftmax_Data4.npy",encoding='latin1').item(); 
	GMFinal = np.load("../results/TRO_Results_Final/D4Diffs/D4Diffs_Data5.npy",encoding='latin1').item(); 
	GreedyFinal = np.load("../results/TRO_Results_Final/D4Diffs/D4Diffs_Data_Greedy4.npy",encoding='latin1').item(); 
	PerfectFinal = np.load("../results/D4Diffs/perfectKnowledgeResults_Stay.npy",encoding='latin1').item(); 


	VBLowQFinal = np.load("../results/D4DiffsSoftmax_Data_LowQ.npy",encoding='latin1').item(); 
	GMLowQFinal = np.load("../results/D4Diffs_Data_LowQ.npy",encoding='latin1').item(); 
	GreedyLowQFinal = np.load("../results/D4Diffs_Data_LowQ_Greedy.npy",encoding='latin1').item(); 
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
		plt.scatter(allX,allY,color=allColors,s=2);
		plt.xlabel("$\Delta$X") 
		plt.ylabel("$\Delta$Y") 
		plt.title(names[r] + " Actions"); 
		plt.savefig('../img/'+names[r]+"Actions.png");
		plt.savefig('../../../../../mnt/c/Users/clbur/OneDrive/Work Docs/Presentations/11_5_18 Nisar Meeting/'+names[r]+'Actions.png'); 



def actionConfusion():
	VBFinal = np.load("../results/TRO_Results_Final/D4DiffsSoftmax/D4DiffsSoftmax_Data4.npy",encoding='latin1').item(); 
	GMFinal = np.load("../results/TRO_Results_Final/D4Diffs/D4Diffs_Data5.npy",encoding='latin1').item(); 
	GreedyFinal = np.load("../results/TRO_Results_Final/D4Diffs/D4Diffs_Data_Greedy4.npy",encoding='latin1').item(); 
	PerfectFinal = np.load("../results/D4Diffs/perfectKnowledgeResults_Stay.npy",encoding='latin1').item(); 


	VBLowQFinal = np.load("../results/D4DiffsSoftmax_Data_LowQ.npy",encoding='latin1').item(); 
	GMLowQFinal = np.load("../results/D4Diffs_Data_LowQ.npy",encoding='latin1').item(); 
	GreedyLowQFinal = np.load("../results/D4Diffs_Data_LowQ_Greedy.npy",encoding='latin1').item(); 
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




if __name__ == '__main__':
	#plotCrossStich(); 
	#extractAndSaveAllData(); 

	#Old
	#makeLayeredPlot(); 
	#makeBoxAndWhisker(); 
	makeHists(); 



	# makeHighQPlots();
	makeLowQPlots(False); 

	#SDExploration(); 

	#actionInvestigation(); 

#	actionConfusion(); 


