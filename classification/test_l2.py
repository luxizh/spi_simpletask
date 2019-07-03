from  snn_l2 import *
import numpy as np
weights=np.load("trainedweight.npy")

for i in range(3):
    spikeTimes=generate_data(i)
    spikes=test(spikeTimes,weights)
    print(i,spikes)
    #print weight_list