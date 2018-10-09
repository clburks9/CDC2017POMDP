import numpy as np
import matplotlib.pyplot as plt 

a = np.load("../results/D2Diffs_2/D2Diffs_2_Timing1.npy"); 
b = np.load("../results/D2Diffs_2Softmax/D2Diffs_2Softmax_Timing1.npy"); 

aC = np.load("../results/D2Diffs_2/D2Diffs_2_Cond_Timing1.npy"); 
bC = np.load("../results/D2Diffs_2Softmax/D2Diffs_2Softmax_Cond_Timing1.npy"); 

# print(a*2);
# print(b*(4/3))



for i in range(0,len(a)):
	if(a[i] > 4*3600):
		print(i,a[i]);
		break;  

for i in range(0,len(b)):
	if(b[i] > 4*3600):
		print(i,b[i]);
		break;  

# print(b-min(b)); 

# print(list(aC))
# print(list(bC)); 




# plt.plot(a); 
# plt.plot(b); 

# plt.show(); 