from __future__ import division

import numpy as np; 
import copy
import matplotlib.pyplot as plt
import sys
from gaussianMixtures import GM,Gaussian
sys.path.append('../src/'); 


#from scipy import stats
#import pyvttbl as pt
from collections import namedtuple

def findMixtureParams(mixture):

	#cut the mixture to just the robber dimensions
	newMixture = GM(); 
	for g in mixture:
		tmpMean = [g.mean[0],g.mean[1]]; 
		tmpVar = [[g.var[0][0],g.var[0][1]],[g.var[1][0],g.var[1][1]]]; 
		newMixture.addG(Gaussian(tmpMean,tmpVar,g.weight)); 

	#cut the mixture to just the robber dimensions

	#mean is a weighted average of means
	mixMean = np.zeros(2);
	for g in newMixture:
		mixMean += np.array(g.mean)*g.weight; 

	#Variance is the weighted sum of variances plus the weighted sum of outer products of the difference of the mean and mixture mean
	mixVar = np.zeros(shape=(2,2)); 
	for g in newMixture:
		mixVar += np.matrix(g.var)*g.weight; 
		mixVar += (np.matrix(g.mean)-np.matrix(mixMean)).T*(np.matrix(g.mean)-np.matrix(mixMean))*g.weight; 

	return mixMean,mixVar;

def heatGrid(data,averageFinalReward):
	grid = [[averageFinalReward['PP'],averageFinalReward['PV']],[averageFinalReward['VP'],averageFinalReward['VV']]]; 
	fig,ax = plt.subplots(); 
	ax.pcolor(grid); 
	ax.set_ylabel('Assumed Robber Dynamics')
	ax.set_xlabel('Actual Robber Dynamics')
	ax.set_xticks([.5,1.5]); 
	ax.set_yticks([.5,1.5]);
	ax.set_xticklabels(('Use NCP','Use NCV'));
	ax.set_yticklabels(('Think NCV','Think NCP')); 
	ax.xaxis.tick_top(); 
	ax.xaxis.set_label_position("top");
	print(averageFinalReward); 
	
	ax.text(0.5,1.5,averageFinalReward['PP'],ha='center',va='center',fontsize=50)
	ax.text(1.5,1.5,averageFinalReward['PV'],ha='center',va='center',fontsize=50)
	ax.text(0.5,0.5,averageFinalReward['VP'],ha='center',va='center',fontsize=50)
	ax.text(1.5,0.5,averageFinalReward['VV'],ha='center',va='center',fontsize=50)
	plt.show();


def fillAndBoxPlots(data,averageFinalReward,averageAllReward,variance,sigma,allSigma):

	UpperBounds = {'PP':[0]*100,'PV':[0]*100,'VP':[0]*100,'VV':[0]*100}; 
	LowerBounds = {'PP':[0]*100,'PV':[0]*100,'VP':[0]*100,'VV':[0]*100}; 

	for key in data.keys():
		for i in range(0,len(data[key]['Rewards'])):
			UpperBounds[key][i] = averageAllReward[key][i]+allSigma[key][i]; 
			LowerBounds[key][i] = averageAllReward[key][i]-allSigma[key][i]; 

	x = [i for i in range(0,100)]; 

	colors = {'PP':'b','PV':'g','VP':'k','VV':'r'}
	leg = []; 
	plt.figure(); 
	for key in data.keys():
		plt.plot(x,averageAllReward[key][0:100],c=colors[key]); 
		plt.plot(x,UpperBounds[key],colors[key]+'--'); 
		plt.plot(x,LowerBounds[key],colors[key]+'--');
		plt.fill_between(x,LowerBounds[key],UpperBounds[key],color=colors[key],alpha=0.25);  
		leg.append(key); 
	plt.legend(leg); 
	plt.xlabel('Time Step'); 
	plt.ylabel('Accumulated Reward')
	plt.title('Average Accumulated Rewards over Time')

	plt.savefig('../results/D4DiffsSoftmax/filledPlot.png',bbox_inches='tight'); 

	fig,ax = plt.subplots(); 
	rects = ax.bar(np.arange(4),[averageFinalReward['PP'],averageFinalReward['PV'],averageFinalReward['VP'],averageFinalReward['VV']],yerr=[sigma['PP'],sigma['PV'],sigma['VP'],sigma['VV']]); 
	for rect in rects:
		height = rect.get_height(); 
		ax.text(rect.get_x() + rect.get_width()/4.,1.05*height, '%d' % int(height), ha='center',va='bottom'); 
	ax.set_xticks(np.arange(4)); 
	ax.set_xticklabels(('PP','PV','VP','VV')); 
	ax.set_ylabel('Average Reward'); 
	ax.set_title('Average Final Rewards for Linear Dynamics Differencing Problem')

	plt.savefig('../results/D4DiffsSoftmax/boxPlot.png',bbox_inches='tight');  




# print(averageFinalReward); 
# print(sigma); 

def anova(data):
	allKeys = ['PP','VP','PV','VV']; 
	N=100;
	P = [1,2]; 
	Q = [1,2]; 
	sub_id = [i+1 for i in xrange(100)]*len(P)*len(Q); 
	reward = np.zeros(shape=(N,len(P),len(Q))).tolist(); 
	for i in range(0,N):
		for j in range(0,len(P)):
			for k in range(0,len(Q)):
				reward[i][j][k] = data[allKeys[j+2*k]]['Rewards'][i][-1]; 

	think = np.concatenate([np.array([p]*N) for p in P]*len(Q)).tolist(); 
	use = np.concatenate([np.array([q]*(N*len(P))) for q in Q]).tolist(); 


	Sub = namedtuple('Sub',['Sub_id','reward','think','use']); 
	df = pt.DataFrame();

	#for idx in xrange(len(sub_id)):
		#print(Sub(sub_id[idx],reward[idx],think[idx],use[idx])._asdict()); 
		#df.insert(Sub(sub_id[idx],reward[idx],think[idx],use[idx])._asdict()); 

	for i in range(0,N):
		for j in range(0,len(P)):
			for k in range(0,len(Q)):
				df.insert(Sub(sub_id[i+j*100+k*200],reward[i][j][k],think[i+j*100+k*200],use[i+j*100+k*200])._asdict()); 

	df.box_plot('reward',factors=['think','use'],fname = '../results/D4DiffsSoftmax/boxAndWhisker.png'); 
	#print(df);
	aov = df.anova('reward',sub='Sub_id',wfactors=['think','use']); 


	print(aov)


def showBoundedRobberEstimate(data):



	dist = {}; 
	for key in data.keys():
		dist[key] = {'means':[],'vars':[]}; 
		
		bels = data[key]['Beliefs'][0];
		XD = data[key]['States(Ind)'][0][0]; 
		YD = data[key]['States(Ind)'][0][1]; 
		#print(len(robs),len(cops),len(bels)); 
		for i in range(0,100):
			mixMean,mixVar = findMixtureParams(bels[i]); 
			dist[key]['means'].append(mixMean); 
			dist[key]['vars'].append([mixVar[0][0],mixVar[1][1]]); 

	
	fig,axarr = plt.subplots(3,4,sharex = True); 
	legend = []; 
	colors = {'NCP/NCP':'b','NCV/NCV':'g','NCP/NCV':'r','NCV/NCP':'m'};
	keys = data.keys(); 

	allKeys = ['NCP/NCP','NCP/NCV','NCV/NCP','NCV/NCV']

	for key in allKeys:
		x = [i for i in range(0,100)]; 
		ind = allKeys.index(key); 


		XD = data[key]['States(Ind)'][0][0][0:100]; 
		YD = data[key]['States(Ind)'][0][1][0:100]; 
		m0 = [dist[key]['means'][i][0] for i in range(0,len(x))]; 
		m1 = [dist[key]['means'][i][1] for i in range(0,len(x))]; 
		howManySigma = 2; 

		upperBound0 = [dist[key]['means'][i][0] + howManySigma*np.sqrt(dist[key]['vars'][i][0]) for i in range(0,len(x))];
		upperBound1 = [dist[key]['means'][i][1] + howManySigma*np.sqrt(dist[key]['vars'][i][1]) for i in range(0,len(x))]; 
		lowerBound0 = [dist[key]['means'][i][0] - howManySigma*np.sqrt(dist[key]['vars'][i][0]) for i in range(0,len(x))]; 
		lowerBound1 = [dist[key]['means'][i][1] - howManySigma*np.sqrt(dist[key]['vars'][i][1]) for i in range(0,len(x))];  

		axarr[0][ind].plot(x,XD,'k--'); 
		axarr[1][ind].plot(x,YD,'k--'); 
		axarr[0][ind].plot(x,m0,c=colors[key]); 
		axarr[1][ind].plot(x,m1,c=colors[key]); 


		axarr[2][ind].plot(x,data[key]['Rewards'][0:100],c=colors[key],linewidth = 3)
		zeros = [0]*100; 
		axarr[2][ind].fill_between(x,zeros,data[key]['Rewards'][0:100],color=colors[key],alpha=0.25)
		axarr[2][ind].legend(["Cumulative Reward"],loc='upper left',fontsize=10); 
		axarr[2][ind].set_ylim([0,125]); 
		axarr[1][ind].set_ylim([-5,5]);
		axarr[0][ind].set_ylim([-5,5]);



		if(ind == 1 or ind == 2 or ind == 3):
			axarr[0][ind].yaxis.set_major_formatter(plt.NullFormatter()); 
			axarr[1][ind].yaxis.set_major_formatter(plt.NullFormatter());  
			#axarr[2][ind].yaxis.set_major_formatter(plt.NullFormatter()); 

		plt.subplots_adjust(left = 0.05,bottom = 0.06,wspace=0.1,hspace=0.06,top=.9,right=0.99); 

		axarr[0][ind].fill_between(x,lowerBound0,upperBound0,color=colors[key],alpha=0.25);
		axarr[1][ind].fill_between(x,lowerBound1,upperBound1,color=colors[key],alpha=0.25); 
		labels = ('True State','Mean',r'2$\sigma$')
		axarr[0][ind].legend(labels,loc='upper right',fontsize=10); 
		axarr[1][ind].legend(labels,loc='upper right',fontsize=10); 

	#allKeys = ['NCP/NCP','NCP/NCV','NCV/NCP','NCV/NCV']
	axarr[0][0].set_ylabel(r'$\Delta$X Estimate',fontsize=20); 
	axarr[1][0].set_ylabel(r'$\Delta$Y Estimate',fontsize=20); 
	axarr[2][0].set_ylabel('Reward',fontsize=20)
	# for i in range(0,len(allKeys)):
	# 	axarr[0][i].set_title(allKeys[i]); 
	axarr[0][0].set_title('NCP Policy, NCP Actual',fontsize=16); 
	axarr[0][1].set_title("NCP Policy, NCV Actual",fontsize=16); 
	axarr[0][2].set_title("NCV Policy, NCP Actual",fontsize=16); 
	axarr[0][3].set_title("NCV Policy, NCV Actual",fontsize=16); 
	axarr[2][0].set_xlabel('Time Step',fontsize=16)
	axarr[2][1].set_xlabel('Time Step',fontsize=16)
	axarr[2][2].set_xlabel('Time Step',fontsize=16)
	axarr[2][3].set_xlabel('Time Step',fontsize=16)
	plt.suptitle("Linear State Dynamics Estimates and Rewards",fontsize=20)

	plt.show(); 

if __name__ == '__main__':


	data = {}; 
	data['NCP/NCP'] = np.load('../results/D4DiffsSoftmax/Good_Old_Data/D4DiffsSoftmax_Data_ThinkNCP_UseNCP.npy',encoding = 'latin1').tolist(); 
	data['NCV/NCV'] = np.load('../results/D4DiffsSoftmax/Good_Old_Data/D4DiffsSoftmax_Data_ThinkNCV_UseNCV.npy',encoding = 'latin1').tolist();
	data['NCP/NCV'] = np.load('../results/D4DiffsSoftmax/Good_Old_Data/D4DiffsSoftmax_Data_ThinkNCP_UseNCV.npy',encoding = 'latin1').tolist();  
	data['NCV/NCP'] = np.load('../results/D4DiffsSoftmax/Good_Old_Data/D4DiffsSoftmax_Data_ThinkNCV_UseNCP.npy',encoding = 'latin1').tolist(); 

	# for i in range(0,len(data['NCV/NCP']['Rewards'])):
	# 	print(i,data['NCV/NCP']['Rewards'][i][-1])

	exampleData = {}; 
	#inds = {'NCP/NCP':25,'NCV/NCV':77,'NCP/NCV':80,'NCV/NCP':92};
	inds = {'NCP/NCP':25,'NCV/NCV':78,'NCP/NCV':80,'NCV/NCP':92};
	for key in inds.keys():
		exampleData[key] = {}; 

	for key in inds.keys():
		for key2 in data['NCP/NCP'].keys():
			exampleData[key][key2] = data[key][key2][inds[key]];
	
	for key in inds.keys():
		print(key,exampleData[key]['Rewards'][-1]); 


	averageFinalReward = {'NCP/NCP':0,'NCP/NCV':0,'NCV/NCP':0,'NCV/NCV':0}; 
	averageAllReward = {'NCP/NCP':[0]*101,'NCP/NCV':[0]*101,'NCV/NCP':[0]*101,'NCV/NCV':[0]*101}; 

	#print(data['NCP/NCP']['Rewards'][0])

	for key in data.keys():
		for i in range(0,len(data[key]['Rewards'])): 
			#print(len(data[key]['Rewards'])); 
			averageFinalReward[key] += data[key]['Rewards'][i][-1]/len(data[key]['Rewards']); 

		for i in range(0,len(data[key]['Rewards'])):
			for j in range(0,len(data[key]['Rewards'][i])):
				#print(len(averageAllReward[key]),len(data[key]['Rewards'][i]),len(data[key]['Rewards'])); 
				averageAllReward[key][j] += data[key]['Rewards'][i][j]/len(data[key]['Rewards']); 


	variance = {'NCP/NCP':0,'NCP/NCV':0,'NCV/NCP':0,'NCV/NCV':0}; 
	sigma = {'NCP/NCP':0,'NCP/NCV':0,'NCV/NCP':0,'NCV/NCV':0}; 
	allSigma = {'NCP/NCP':[0]*101,'NCP/NCV':[0]*101,'NCV/NCP':[0]*101,'NCV/NCV':[0]*101}; 


	for key in data.keys():
		suma = 0; 
		for i in range(0,len(data[key]['Rewards'])):
			suma+=(data[key]['Rewards'][i][-1] - averageFinalReward[key])**2; 
		variance[key] = suma/len(data[key]['Rewards']); 
		sigma[key] = np.sqrt(variance[key]); 

		for i in range(0,len(data[key]['Rewards'][0])):
			suma = 0; 
			for j in range(0,len(data[key]['Rewards'])):
				suma += (data[key]['Rewards'][j][i] - averageAllReward[key][i])**2; 
			allSigma[key][i] = np.sqrt(suma/len(data[key]['Rewards'])); 

	print(averageFinalReward)
	print(sigma); 


	#heatGrid(data,averageFinalReward);
	#fillAndBoxPlots(data,averageFinalReward,averageAllReward,variance,sigma,allSigma);

	#showBoundedRobberEstimate(exampleData); 












