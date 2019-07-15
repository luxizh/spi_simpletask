import pyNN.spiNNaker as sim
import numpy as np
import random
from pyNN.random import NumpyRNG, RandomDistribution
import pyNN.utility.plotting as pplt
import matplotlib.pyplot as plt

#======stdp network========
sim.setup(timestep=1)

pre_spikes=[]
post_spikes=[]
v=40

for i in range(10):
    post_spikes.append(i*v+5)
for i in range(20):
    pre_spikes.append(i*v)
    #pre_spikes.append(i*v+2)


runTime=(i+1)*v
cell_params_lif = {'cm': 1,#70
                   'i_offset': 0.0,
                   'tau_m': 20.0,#20
                   'tau_refrac': 10.0,#2 more that t inhibit#10
                   'tau_syn_E': 2.0,#2
                   'tau_syn_I': 10.0,#5
                   'v_reset': -70.0,
                   'v_rest': -65.0,
                   'v_thresh': -55.0
                   }

pre_cell=sim.SpikeSourceArray(pre_spikes)
post_cell=sim.SpikeSourceArray(post_spikes)
layer1=sim.Population(1,pre_cell,label='inputspikes')
post=sim.Population(1,sim.IF_curr_exp,cellparams=cell_params_lif,label='post')
layer2=sim.Population(1,post_cell,label='outputspikes')



stim_proj = sim.Projection(layer2, post, sim.OneToOneConnector(), 
                            synapse_type=sim.StaticSynapse(weight=7, delay=0.25))


stdp = sim.STDPMechanism(
                        weight=0,
                        #weight=0.02,  # this is the initial value of the weight
                        #delay="0.2 + 0.01*d",
                        timing_dependence=sim.SpikePairRule(tau_plus=30, tau_minus=10,A_plus=0.5, A_minus=0.5),
                        #weight_dependence=sim.MultiplicativeWeightDependence(w_min=wMin, w_max=wMax),
                        weight_dependence=sim.AdditiveWeightDependence(w_min=0, w_max=5),
                        #weight_dependence=sim.AdditiveWeightDependence(w_min=0, w_max=0.4),
                        dendritic_delay_fraction=1.0)

stdp_proj = sim.Projection(layer1, post, sim.AllToAllConnector(), synapse_type=stdp)

layer1.record(['spikes'])
layer2.record(['spikes'])
post.record(['v','spikes'])

sim.run(runTime)
weight_list = [stdp_proj.get('weight', 'list'), stdp_proj.get('weight', format='list', with_address=False)]

neo = post.get_data(["spikes", "v"])
spikes = neo.segments[0].spiketrains
v = neo.segments[0].filter(name='v')[0]
neoinput= layer1.get_data(["spikes"])
spikesinput = neoinput.segments[0].spiketrains
neooutput= layer2.get_data(["spikes"])
spikesoutput = neooutput.segments[0].spiketrains

plt.close('all')
pplt.Figure(
pplt.Panel(v, ylabel="Membrane potential (mV)", xticks=True, yticks=True, xlim=(0,runTime)),
pplt.Panel(spikesinput, xticks=True, yticks=True, markersize=2, xlim=(0,runTime)),
pplt.Panel(spikesoutput, xticks=True, yticks=True, markersize=2, xlim=(0,runTime)),
#pplt.Panel(spikestim, xticks=True, yticks=True, markersize=2, xlim=(0,runTime)),
pplt.Panel(spikes, xticks=True, xlabel="Time (ms)", yticks=True, markersize=2, xlim=(0,runTime)),
title="Training",
annotations="Training"
            ).save('testplot/training4.png')


sim.end()



