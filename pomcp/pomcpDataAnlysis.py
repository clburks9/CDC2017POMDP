from __future__ import division
import numpy as np
import matplotlib.pyplot as plt


def fillAndBoxPlots(data,averageFinalReward,averageAllReward,variance,sigma,allSigma,fill=True,box=True):

	if(fill):
		UpperBounds = {'one':[0]*100,'two':[0]*100,'three':[0]*100,'one_s':[0]*100}; 
		LowerBounds = {'one':[0]*100,'two':[0]*100,'three':[0]*100,'one_s':[0]*100}; 

		for key in data.keys():
			for i in range(0,len(data[key])):
				UpperBounds[key][i] = averageAllReward[key][i]+allSigma[key][i]; 
				LowerBounds[key][i] = averageAllReward[key][i]-allSigma[key][i]; 

		x = [i for i in range(0,100)]; 

		colors = {'one':'b','two':'g','three':'r','one_s':'k'}
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

		plt.savefig('../pomcp/filledPlot.png',bbox_inches='tight'); 

	if(box):
		fig,ax = plt.subplots(); 
		rects = ax.bar(np.arange(3),[averageFinalReward['one'],averageFinalReward['two'],averageFinalReward['three']],yerr=[sigma['one'],sigma['two'],sigma['three']]); 
		for rect in rects:
			height = rect.get_height(); 
			ax.text(rect.get_x() + rect.get_width()/3.,1.05*height, '%d' % int(height), ha='center',va='bottom'); 
		ax.set_xticks(np.arange(3)); 
		ax.set_xticklabels(('one','two','three')); 
		ax.set_ylabel('Average Reward'); 
		ax.set_title('Average Final Rewards for Linear Dynamics Differencing Problem')

		plt.savefig('../pomcp/boxPlot.png',bbox_inches='tight');  


if __name__ == '__main__':
	

	data = {'one':[],'two':[],'three':[]}; 


	for key in data.keys():
		with open("{}SecondPOMCP.txt".format(key)) as f:
			for line in f:
				slines = line.split(); 
				intLines = []; 
				for i in range(0,len(slines)):
					intLines.append(int(slines[i])); 
				data[key].append(intLines);

	data['one_s'] = []; 

	with open("oneSecondPOMCP_stationary.txt") as f: 
		for line in f:
			slines = line.split(); 
			intLines = []; 
			for i in range(0,len(slines)):
				intLines.append(int(slines[i])); 
			data['one_s'].append(intLines); 


	averageFinalReward = {'one':0,'two':0,'three':0,'one_s':0}; 
	averageAllReward = {'one':[0]*101,'two':[0]*101,'three':[0]*101,'one_s':[0]*101}; 


	for key in data.keys():
		for i in range(0,len(data[key])): 
			#print(data[key][i][-1]); 
			averageFinalReward[key] += data[key][i][-1]/len(data[key]); 

		for i in range(0,len(data[key])):
			for j in range(0,len(data[key][i])):
				#print(len(averageAllReward[key]),len(data[key][i]),len(data[key])); 
				averageAllReward[key][j] += data[key][i][j]/len(data[key]); 



	variance = {'one':0,'two':0,'three':0,'one_s':0}; 
	sigma = {'one':0,'two':0,'three':0,'one_s':0}; 
	allSigma = {'one':[0]*101,'two':[0]*101,'three':[0]*101,'one_s':[0]*101}; 


	for key in data.keys():
		suma = 0; 
		for i in range(0,len(data[key])):
			suma+=(data[key][i][-1] - averageFinalReward[key])**2; 
		variance[key] = suma/len(data[key]); 
		sigma[key] = np.sqrt(variance[key]); 

		for i in range(0,len(data[key][0])):
			suma = 0; 
			for j in range(0,len(data[key])):
				suma += (data[key][j][i] - averageAllReward[key][i])**2; 
			allSigma[key][i] = np.sqrt(suma/len(data[key])); 



	#print(averageFinalReward); 
	#print(sigma); 
	fillAndBoxPlots(data,averageFinalReward,averageAllReward,variance,sigma,allSigma,box=False);



