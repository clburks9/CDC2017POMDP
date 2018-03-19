from __future__ import division

import numpy as np; 
import copy
import matplotlib.pyplot as plt
import sys
sys.path.append('../src/'); 


#from scipy import stats
import pyvttbl as pt
from collections import namedtuple


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

if __name__ == '__main__':


	data = {}; 
	data['PP'] = np.load('../results/D4DiffsSoftmax/D4DiffsSoftmax_Data_ThinkNCP_UseNCP.npy').tolist(); 
	data['VV'] = np.load('../results/D4DiffsSoftmax/D4DiffsSoftmax_Data_ThinkNCV_UseNCV.npy').tolist();
	data['PV'] = np.load('../results/D4DiffsSoftmax/D4DiffsSoftmax_Data_ThinkNCP_UseNCV.npy').tolist();  
	data['VP'] = np.load('../results/D4DiffsSoftmax/D4DiffsSoftmax_Data_ThinkNCV_UseNCP.npy').tolist(); 

	averageFinalReward = {'PP':0,'PV':0,'VP':0,'VV':0}; 
	averageAllReward = {'PP':[0]*101,'PV':[0]*101,'VP':[0]*101,'VV':[0]*101}; 


	for key in data.keys():
		for i in range(0,len(data[key]['Rewards'])): 
			#print(len(data[key]['Rewards'])); 
			averageFinalReward[key] += data[key]['Rewards'][i][-1]/len(data[key]['Rewards']); 

		for i in range(0,len(data[key]['Rewards'])):
			for j in range(0,len(data[key]['Rewards'][i])):
				#print(len(averageAllReward[key]),len(data[key]['Rewards'][i]),len(data[key]['Rewards'])); 
				averageAllReward[key][j] += data[key]['Rewards'][i][j]/len(data[key]['Rewards']); 


	variance = {'PP':0,'PV':0,'VP':0,'VV':0}; 
	sigma = {'PP':0,'PV':0,'VP':0,'VV':0}; 
	allSigma = {'PP':[0]*101,'PV':[0]*101,'VP':[0]*101,'VV':[0]*101}; 


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



	heatGrid(data,averageFinalReward);
	fillAndBoxPlots(data,averageFinalReward,averageAllReward,variance,sigma,allSigma);













