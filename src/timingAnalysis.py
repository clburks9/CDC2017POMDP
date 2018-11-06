
import numpy as np
import matplotlib.pyplot as plt


GMtimes = np.load("../results/D4Diffs/D4Diffs_TimingNCP.npy"); 
VBtimes = np.load("../results/D4DiffsSoftmax/D4DiffsSoftmax_TimingNCP.npy"); 

CGMtimes = np.load("../results/D4Diffs/D4Diffs_Cond_TimingNCP.npy");
CVBtimes = np.load("../results/D4DiffsSoftmax/D4DiffsSoftmax_Cond_TimingNCP.npy"); 



#Find the half hourly policies
GMHours = {}; 
VBHours = {}; 
for i in range(1,8*4+1):
	for j in range(0,len(GMtimes)):
		if(GMtimes[j] > i*3600/4):
			GMHours[str(i/4)] = j-1; 
			break; 
	for j in range(0,len(VBtimes)):
		if(VBtimes[j] > i*3600/4):0
			VBHours[str(i/4)] = j-1; 
			break; 

print(sorted(GMHours.items())); 
print(sorted(VBHours.items())); 


diffsGM = []; 
for i in range(1,len(GMtimes)):
	diffsGM.append(GMtimes[i]-GMtimes[i-1])

diffsVB = []; 
for i in range(1,len(VBtimes)):
	diffsVB.append(VBtimes[i]-VBtimes[i-1])

plt.plot(diffsGM); 
plt.plot(diffsVB)


plt.show(); 