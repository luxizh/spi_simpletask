import pyNN.spiNNaker as sim

sim.setup(timestep=0.1)
layer1=sim.Population(50,sim.IF_cond_exp,
                    cellparams={'i_offset':0.11,'tau_refrac':3.0,'v_thresh':-51.0},
                    label='layer1')
stdp = sim.STDPMechanism(
                        weight=0.02,  # this is the initial value of the weight
                        #delay="0.2 + 0.01*d",
                        timing_dependence=sim.SpikePairRule(tau_plus=20.0, tau_minus=20.0,A_plus=0.01, A_minus=0.012),
                        weight_dependence=sim.AdditiveWeightDependence(w_min=0, w_max=0.04),
                        dendritic_delay_fraction=0)
sim.end()
