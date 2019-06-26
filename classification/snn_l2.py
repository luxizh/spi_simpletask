import pyNN.spiNNaker as sim
import numpy as np
import random
from pyNN.random import NumpyRNG, RandomDistribution

#def parameters
__delay__ = 1.0 # (ms) 
tauPlus = 30 #20 # 15 # 16.8 from literature
tauMinus = 30 #20 # 30 # 33.7 from literature
aPlus = 0.500  #tum 0.016 #9 #3 #0.5 # 0.03 from literature
aMinus = 0.500 #255 #tum 0.012 #2.55 #2.55 #05 #0.5 # 0.0255 (=0.03*0.85) from literature 
wMax = 10.000 #1 # G: 0.15
wMaxInit = 10.000#0.1#0.100
wMin = 0
nbIter = 5
testWeightFactor = 1#0.05177
x = 3 # no supervision for x first traj presentations
y = 0# for inside testing of traj to see if it has been learned /!\ stdp not disabled

input_len=3
input_size=input_len*input_len
output_size=3
inhibWeight = 10
stimWeight = 100

v_co=3

cell_params_lif = {'cm': 0.25,
                   'i_offset': 0.0,
                   'tau_m': 20.0,
                   'tau_refrac': 2.0,
                   'tau_syn_E': 5.0,
                   'tau_syn_I': 5.0,
                   'v_reset': -70.0,
                   'v_rest': -65.0,
                   'v_thresh': -50.0
                   }

def generate_data(label):
    spikesTrain=[]
    organisedData = {}
    for i in range(input_len):
        for j in range(input_len):
            neuid=(i,j)
            organisedData[neuid]=[]
    for i in range(input_len):
        neuid=(i,label)
        organisedData[neuid].append(i*v_co)
#        if neuid not in organisedData:
#            organisedData[neuid]=[i*v_co]
#        else:
#            organisedData[neuid].append(i*v_co)
    
    for neuronSpikes in organisedData.values():
        neuronSpikes.sort()
        spikesTrain.append(neuronSpikes)
    return spikesTrain

def train(label,untrained_weights=None):
    organisedStim = {}
    labelSpikes = []
    spikeTimes = generate_data(label)

    for i in range(output_size):
        labelSpikes.append([])
    labelSpikes[label] = [int(max(max(spikeTimes)))+1]
    
    
    if untrained_weights == None:
        untrained_weights = RandomDistribution('uniform', low=wMin, high=wMaxInit).next(input_size*output_size)
        #untrained_weights = RandomDistribution('normal_clipped', mu=0.1, sigma=0.05, low=wMin, high=wMaxInit).next(input_size*output_size)
        untrained_weights = np.around(untrained_weights, 3)
        #saveWeights(untrained_weights, 'untrained_weightssupmodel1traj')

    print "length untrained_weights :", len(untrained_weights)

    if len(untrained_weights)>input_size:
        training_weights = [[0 for j in range(output_size)] for i in range(input_size)] #np array? size 1024x25
        k=0
        for i in range(input_size):
            for j in range(output_size):
                training_weights[i][j] = untrained_weights[k]
                k += 1
    else:
        training_weights = untrained_weights[0]

    connections = []
    for n_pre in range(input_size): # len(untrained_weights) = input_size
        for n_post in range(output_size): # len(untrained_weight[0]) = output_size; 0 or any n_pre
            connections.append((n_pre, n_post, training_weights[n_pre][n_post], __delay__)) 
    runTime = int(max(max(spikeTimes)))+1000
    #####################
    sim.setup(timestep=1)
    #def populations
    layer1=sim.Population(input_size,sim.SpikeSourceArray, {'spike_times': spikeTimes},label='inputspikes')
    layer2=sim.Population(output_size,sim.IF_curr_exp,cellparams=cell_params_lif,label='outputspikes')
    supsignal=sim.Population(output_size,sim.SpikeSourceArray, {'spike_times': labelSpikes},label='supersignal')

    #def learning rule
    stdp = sim.STDPMechanism(
                            weight=0.02,  # this is the initial value of the weight
                            #delay="0.2 + 0.01*d",
                            timing_dependence=sim.SpikePairRule(tau_plus=20.0, tau_minus=20.0,A_plus=0.01, A_minus=0.012),
                            weight_dependence=sim.MultiplicativeWeightDependence(w_min=wMin, w_max=wMax),
                            #weight_dependence=sim.AdditiveWeightDependence(w_min=0, w_max=0.4),
                            dendritic_delay_fraction=1.0)
    #def projections

    stdp_proj = sim.Projection(layer1, layer2, sim.FromListConnector(connections), synapse_type=stdp)
    inhibitory_connections = sim.Projection(layer2, layer2, sim.AllToAllConnector(allow_self_connections=False), 
                                            synapse_type=sim.StaticSynapse(weight=inhibWeight, delay=__delay__), 
                                            receptor_type='inhibitory')
    stim_proj = sim.Projection(supsignal, layer2, sim.OneToOneConnector(), 
                                synapse_type=sim.StaticSynapse(weight=stimWeight, delay=__delay__))

    #layer2.record(['v', 'spikes'])
    #supsignal.record(['spikes'])
    sim.run(runTime)

    print("Weights:{}".format(stdp_proj.get('weight', 'list')))

    weight_list = [stdp_proj.get('weight', 'list'), stdp_proj.get('weight', format='list', with_address=False)] 
                    
    sim.end()
    return weight_list


def test(spikeTimes, trained_weights):

    #spikeTimes = extractSpikes(sample)
    runTime = int(max(max(spikeTimes)))+1000

    ##########################################

    sim.setup(timestep=1)

    pre_pop = sim.Population(input_size, sim.SpikeSourceArray, {'spike_times': spikeTimes}, label="pre_pop")
    post_pop = sim.Population(output_size, sim.IF_curr_exp , cell_params_lif, label="post_pop")
   
    if len(trained_weights) > input_size:
        weigths = [[0 for j in range(output_size)] for i in range(input_size)] #np array? size 1024x25
        k=0
        for i in range(input_size):
            for j in range(output_size):
                weigths[i][j] = trained_weights[k]
                k += 1
    else:
        weigths = trained_weights

    connections = []
    
    #k = 0
    for n_pre in range(input_size): # len(untrained_weights) = input_size
        for n_post in range(output_size): # len(untrained_weight[0]) = output_size; 0 or any n_pre
            connections.append((n_pre, n_post, weigths[n_pre][n_post]*(wMax)/max(trained_weights), __delay__)) #
            #k += 1

    prepost_proj = sim.Projection(pre_pop, post_pop, sim.FromListConnector(connections), synapse_type=sim.StaticSynapse(), receptor_type='excitatory') # no more learning !!
    #inhib_proj = sim.Projection(post_pop, post_pop, sim.AllToAllConnector(), synapse_type=sim.StaticSynapse(weight=inhibWeight, delay=__delay__), receptor_type='inhibitory')
    # no more lateral inhib

    post_pop.record(['v', 'spikes'])
    sim.run(runTime)

    neo = post_pop.get_data(['v', 'spikes'])
    spikes = neo.segments[0].spiketrains
    v = neo.segments[0].filter(name='v')[0]

    print("Weights:{}".format(prepost_proj.get('weight', 'list')))

    weight_list = [prepost_proj.get('weight', 'list'), prepost_proj.get('weight', format='list', with_address=False)]
    #predict_label=
    sim.end()


#==============main================
weight_list=None

for i in range(10):
    label=random.randint(0,2)
    weight_list=train(label=label,untrained_weights=weight_list)
    #weight_list=weight_list[0]
    print weight_list


for i in range(3):
    spikeTimes=generate_data(i)
    test(spikeTimes,weight_list)

