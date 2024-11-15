import os
import numpy as np

mouse_names = ['L1_A5', 'L1_A1',  'FX9', 'FX10', 'FX8', 'FX20', 'FX40']
NNs = [6636,6055,3575,4792,5804,2746, 4261]

def main():
    mouse_id = 5
    for mouse_id in [6]:
        # for nconv in [64]:
        nconv1 = 192
        nconv2 = 192
        seed = 1
        nlayers = 2
        n_max_neurons = NNs[mouse_id] # Total number of neurons
        # n_max_stims = 4640 # Total number of unique train stimuli
        n_neuron = -1 # Number of neurons to sample
        n_stim_train = -1 # 5000 # -1
        weight_decay_core = 0.1
        pretrain_mouse_id = -100 # -1 for use the same mouse, -100 for no pretraining

        hs_list = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.5]
        conv1_ks_list = [7,13,17,21,25,29]
        conv2_ks_list = [5,7,9,11,13,15]

        conv1_ks = 25
        conv2_ks = 9
        hs_readout = 0.0
        # for pretrain_mouse_id in [-100, -1]:
        #     for n_stim_train in [30000, 5000]:
        # for conv1_ks in conv1_ks_list:
        #     for conv2_ks in conv2_ks_list:
        output_save_path = f'outputs/fullmodel/{mouse_names[mouse_id]}'
        if not os.path.exists(output_save_path):
            os.makedirs(output_save_path)

        # nconv1_list = [8,16,32,64, 128, 192, 256, 320, 384, 448]
        # nconv2_list = [8,16,32,64, 128, 192, 256, 320, 384, 448]

        # for nconv1 in nconv1_list:
        #     for nconv2 in nconv2_list:
        
        for nlayers in range(1,5):

        # Generate lists of neuron numbers and seed numbers using logarithmic spacing
        # neuron_numbers = np.geomspace(1, 1000, num=10, dtype=int)
        # neuron_numbers = np.unique(np.concatenate(([1], neuron_numbers)))  # Ensure 1 is included and remove duplicates
        # seed_numbers = np.linspace(10, 1, num=len(neuron_numbers), dtype=int)
        # for i, n_neuron in enumerate(neuron_numbers):
        #     for seed in range(1, seed_numbers[i]+1):
            
                # stim_numbers = np.geomspace(500, 30000, num=10, dtype=int)
                # stim_numbers = np.unique(stim_numbers)  # Remove duplicates that might occur due to rounding
                # stim_numbers = [5000, 30000]
                # for n_stim_train in stim_numbers:
        # for hs_readout in hs_list:
            prefix = f'fullmodel_{mouse_names[mouse_id]}_{nlayers}_{nconv1}_{nconv2}_seed{seed}'
            bsub_cmd = f'bsub -n 2 -q gpu_a100 -gpu "num=1"  -J {prefix} -o {output_save_path}/{prefix}.out -e {output_save_path}/{prefix}.err "bash fullmodel_mouse_script_a100.sh {nlayers} {nconv1} {nconv2} {seed} {n_neuron} {n_stim_train} {weight_decay_core} {mouse_id} {conv1_ks} {conv2_ks} {pretrain_mouse_id} {hs_readout}"'
            print(bsub_cmd)
            os.system(bsub_cmd)

if __name__=='__main__':
    main()