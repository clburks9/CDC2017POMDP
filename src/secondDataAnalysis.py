from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from gaussianMixtures import GM,Gaussian



def extractOldData():
	data = {};

	data['GM'] = np.load('../results/D4Diffs/D4Diffs_Data_ThinkNCP_UseNCP2.npy',encoding='latin1').tolist(); 
	data['VB'] = np.load('../results/D4DiffsSoftmax/D4DiffsSoftmax_Data_ThinkNCP_UseNCP3.npy',encoding='latin1').tolist(); 
	data['Greedy'] = np.load('../results/D4Diffs/D4Diffs_Data_ThinkNCP_UseNCP_Greedy2.npy',encoding='latin1').tolist(); 

	#print(data['GM'].keys()); 
	keys = ['GM','VB','Greedy']; 


	newData = {}; 
	

	#keyNums = {'GM':19,'VB':78,'Greedy':22}; 
	#keyNums = {'GM':27,'VB':6,'Greedy':20}; 
	keyNums = {'GM':27,'VB':78,'Greedy':20}; 

	newData['GM'] = {}; 
	newData['VB'] = {}; 
	newData['Greedy'] = {}; 

	for key in keys:
		for key2 in data['GM'].keys():
			newData[key][key2] = data[key][key2][keyNums[key]];

	file = '../results/ExampleData.npy'; 
	np.save(file,newData); 



def getDistances(data,vis=False):
	dist = {}; 
	for key in data.keys():
		dist[key] = []; 
		xd = data[key]['States(Ind)'][0][0]; 
		yd = data[key]['States(Ind)'][0][1]; 
		#print(key,xd[0])
		# for r in runs: 
		# 	dist[key][str(r)] = []; 
		# 	for i in range(0,min(len(cp[r]),len(rp[r]))):
		# 		dist[key][str(r)].append(np.sqrt((cp[r][i][0]-rp[r][i][0])**2 + (cp[r][i][1]-rp[r][i][1])**2)); 

		for i in range(0,min(len(xd),len(yd))):
			dist[key].append(np.sqrt(xd[i]**2 + yd[i]**2)); 

	if(vis):
		legend = [];
		colors = ['b','g','r']; 
		for key in dist.keys():
			legend.append(key); 
			plt.plot(dist[key]); 
		plt.legend(legend); 
		plt.show(); 
	else:
		return dist;


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

	
	fig,axarr = plt.subplots(3,3,sharex = True); 
	#fig,axarr = plt.subplots(2,3); 
	#fig.text(0.5,0.04,'Seconds',ha='center'); 
	legend = []; 
	colors = {'GM':'b','VB':'g','Greedy':'r'};
	keys = data.keys(); 

	allKeys = ['GM','VB','Greedy']

	for key in allKeys:
		#secondsPerStep = runTimes[run][key]/len(dist[key]['means']);
		#secondsPerStep=1; 
		x = [i for i in range(0,100)]; 
		ind = allKeys.index(key); 

		# error0 = [dist[key]['means'][i][0] - data[key]['RobberPose'][run][i][0] for i in range(0,len(x))]; 
		# error1 = [dist[key]['means'][i][1] - data[key]['RobberPose'][run][i][1] for i in range(0,len(x))];
		XD = data[key]['States(Ind)'][0][0][0:100]; 
		YD = data[key]['States(Ind)'][0][1][0:100]; 
		m0 = [dist[key]['means'][i][0] for i in range(0,len(x))]; 
		m1 = [dist[key]['means'][i][1] for i in range(0,len(x))]; 
		howManySigma = 2; 

		upperBound0 = [dist[key]['means'][i][0] + howManySigma*np.sqrt(dist[key]['vars'][i][0]) for i in range(0,len(x))];
		upperBound1 = [dist[key]['means'][i][1] + howManySigma*np.sqrt(dist[key]['vars'][i][1]) for i in range(0,len(x))]; 
		lowerBound0 = [dist[key]['means'][i][0] - howManySigma*np.sqrt(dist[key]['vars'][i][0]) for i in range(0,len(x))]; 
		lowerBound1 = [dist[key]['means'][i][1] - howManySigma*np.sqrt(dist[key]['vars'][i][1]) for i in range(0,len(x))];  

		#axarr[0][ind].plot(x,XD,colors[key]+'--'); 
		axarr[0][ind].plot(x,XD,'k--'); 
		#axarr[1][ind].plot(x,YD,colors[key]+'--'); 
		axarr[1][ind].plot(x,YD,'k--'); 
		axarr[0][ind].plot(x,m0,c=colors[key]); 
		axarr[1][ind].plot(x,m1,c=colors[key]); 


		axarr[2][ind].plot(x,data[key]['Rewards'][0:100],c=colors[key],linewidth = 3)
		zeros = [0]*100; 
		axarr[2][ind].fill_between(x,zeros,data[key]['Rewards'][0:100],color=colors[key],alpha=0.25)
		axarr[2][ind].legend(["Cumulative Reward"],loc='upper left',fontsize=10); 
		axarr[2][ind].set_ylim([0,125]); 
		axarr[1][ind].set_ylim([-10,10]);
		axarr[0][ind].set_ylim([-10,10]);



		if(ind == 1 or ind == 2):
			axarr[0][ind].yaxis.set_major_formatter(plt.NullFormatter()); 
			axarr[1][ind].yaxis.set_major_formatter(plt.NullFormatter());  
			#axarr[2][ind].yaxis.set_major_formatter(plt.NullFormatter()); 

		plt.subplots_adjust(left = 0.05,bottom = 0.06,wspace=0.1,hspace=0.06,top=.9,right=0.99); 

		axarr[0][ind].fill_between(x,lowerBound0,upperBound0,color=colors[key],alpha=0.25);
		axarr[1][ind].fill_between(x,lowerBound1,upperBound1,color=colors[key],alpha=0.25); 
		labels = ('True State','Mean',r'2$\sigma$')
		axarr[0][ind].legend(labels,fontsize=10,loc='upper right'); 
		axarr[1][ind].legend(labels,fontsize=10,loc='upper right'); 

		# zeros = [0]*100; 
		# distance = [];
		# meanDist = [];
		# upperDist = [];
		# lowerDist = []; 
		# for i in range(0,100):
		# 	distance.append(np.sqrt(XD[i]**2 + YD[i]**2)); 
		# 	meanDist.append(np.sqrt(m0[i]**2 + m1[i]**2))
		# 	upperDist.append((np.sqrt(dist[key]['vars'][i][0] + np.sqrt(dist[key]['vars'][i][0]))*2 + meanDist[-1]))
			# upperDist.append((np.sqrt(dist[key]['vars'][i][0] + np.sqrt(dist[key]['vars'][i][0]))*2))
			# lowerDist.append(-(np.sqrt(dist[key]['vars'][i][0] + np.sqrt(dist[key]['vars'][i][0]))*2))

		# error = []; 
		# for i in range(0,100):
		# 	error.append(meanDist[i]-distance[i]);

		#axarr[0][ind].plot(x,meanDist,c=colors[key]); 
		#axarr[0][ind].plot(x,distance,colors[key]+'-.'); 

		# axarr[0][ind].plot(x,error,c=colors[key]); 
 


		#axarr[0][ind].fill_between(x,zeros,upperDist,color=colors[key],alpha=0.25)
		#axarr[0][ind].fill_between(x,zeros,lowerDist,color=colors[key],alpha=0.25)

		#axarr[1][ind].plot(x,data[key]['Rewards'][0:100],c=colors[key])
		#axarr[1][ind].fill_between(x,zeros,data[key]['Rewards'][0:100],color=colors[key],alpha=0.25)
		#axarr[1][ind].set_ylim([0,125]); 



		# axarr[ind][0].set_ylabel(key); 


	axarr[0][0].set_ylabel(r'$\Delta$X Estimate',fontsize=20); 
	axarr[1][0].set_ylabel(r'$\Delta$Y Estimate',fontsize=20); 
	axarr[2][0].set_ylabel('Reward',fontsize=20)
	axarr[0][0].set_title('GM-POMDP',fontsize=20); 
	axarr[0][1].set_title("VB-POMDP",fontsize=20); 
	axarr[0][2].set_title("Greedy",fontsize=20); 
	axarr[2][0].set_xlabel('Time Step',fontsize=20)
	axarr[2][1].set_xlabel('Time Step',fontsize=20)
	axarr[2][2].set_xlabel('Time Step',fontsize=20)
	plt.suptitle("2D Target Search Estimates and Rewards",fontsize=25)


	

	

	# fig = plt.gcf(); 
	# fig.set_size_inches(10,8); 

	# if(mapType=='a'):
	# 	plt.savefig('./figures/mapA/{}_positionEstimates.png'.format(runKeys[run]),bbox_inches='tight',pad_inches=0,dpi=300);
	# else:
	# 	plt.savefig('./figures/mapC/{}_positionEstimates.png'.format(runKeys[run]),bbox_inches='tight',pad_inches=0,dpi=300);
	plt.show(); 


def malDist(mean,cov,point):
	return np.sqrt((np.matrix(point)-np.matrix(mean))*np.matrix(cov).I*(np.matrix(point)-np.matrix(mean).T));

def getDistOfSTD(mean,cov,point):
	#taking the slice of the belief along the line between the truth and the mean
	#finding for a given timestep the standard deviation

	#find points along the distance from mean to point
	slope = (point[1]-mean[1])/(point[0]-mean[0]); 
	bias = slope*mean[0] - mean[1]; 

	#test 20 points, evenly between mean and point
	if(mean[0] < point[0]):
		testPoints = [[i/10,(i/10*slope+bias)] for i in range(mean[0]*10,point[0]*10)];
	else:
		testPoints = [[i/10,(i/10*slope+bias)] for i in range(point[0]*10,mean[0]*10)];

	print(testPoints);

	#test those points and grab the one closest to a single sigma

	#find it's distance and return it





def distanceBoundsTest():

	var1 = 2; 
	var2 = 1; 
	offDiag = 0;

	g = GM(); 
	g.addG(Gaussian([0,0],[[var1,offDiag],[offDiag,var2]],1)); 

	[x,y,c] = g.plot2D(low=[-5,-5],high=[5,5],vis=False);

	valAtStd = g.pointEval([np.sqrt(var1),np.sqrt(var2)]); 
	print("STD Value: {}".format(valAtStd))
	err = 0.01; 
	allInds = [[],[]];
	for i in range(0,len(c)):
		for j in range(0,len(c[i])):
			if(c[i][j] > valAtStd-err and c[i][j] < valAtStd+err):
				allInds[0].append(i/10 - 5); 
				allInds[1].append(j/10 - 5); 

	#dist = np.sqrt(6**2+7**2); 
	#print("Mean Distance:{}".format(dist))
	aveDist = 0; 
	for i in range(0,len(allInds)):
		aveDist += np.sqrt((allInds[0][i])**2 + (allInds[1][i])**2)/len(allInds); 
	print("Average Deviation from Mean Dist: {}".format(aveDist))

	fig,ax = plt.subplots(); 

	ax.contourf(x,y,c); 
	ax.scatter(allInds[0],allInds[1],c='r'); 
	ax.set_xlim([-5,5]); 
	ax.set_ylim([-5,5])
	plt.show(); 



	



if __name__ == '__main__':
	

	#extractOldData(); 
	data = np.load('../results/ExampleData.npy').tolist(); 
	#print(data['Greedy']['Rewards'][-1])

	#print(data['GM']['Rewards']); 
	# for key in data.keys():
	# 	print(key,data[key]['Rewards'][-1])

	showBoundedRobberEstimate(data)
	#distanceBoundsTest(); 


	#getDistOfSTD([1,1],[[1,0],[0,1]],[3,2]); 
