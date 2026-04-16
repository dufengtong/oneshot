import os
import numpy as np

mouse_names = ['TX104', 'TX110', 'TX80', 'TX91', 'TX115', 'TX114']

def main():
    mouse_id = 2
    for mouse_id in [5]:
        nconv1 = 16
        nconv2 = 320
        seed = 1
        nlayers = 2
        # n_max_neurons = NNs[mouse_id] # Total number of neurons
        # n_max_stims = 4640 # Total number of unique train stimuli
        n_neuron = -1 # Number of neurons to sample
        n_stim_train = -1
        # weight_decay_core = 0.1
        gpu = 'a100'
        area = 0 # 0:all, 1:v1, 2:PM
        pretrain_mouse_id = -100 # -100

        lrs = [0, 0.006, 0, 0]
        weight_decay_cores = [0, 0.1, 0, 0]
        l2_readouts = [0, 0.01, 0, 0]

        output_save_path = f'outputs/fullmodel/{mouse_names[mouse_id]}'
        if not os.path.exists(output_save_path):
            os.makedirs(output_save_path)

        # nconv1_list = [2,4,8,16,32,64, 128, 192, 256, 320, 384, 448]
        # nconv2_list = [2,4,8,16,32,64, 128, 192, 256, 320, 384, 448]

        # for nconv1 in nconv1_list:
        #     for nconv2 in nconv2_list:
        
        # for nlayers in range(1,5):
        #     if nlayers == 1: nconv1 = 192
        #     else: nconv1 = 16
            
        weight_decay_core = weight_decay_cores[nlayers-1]
        lr = lrs[nlayers-1]
        l2_readout = l2_readouts[nlayers-1]

        # Generate lists of neuron numbers and seed numbers using logarithmic spacing
        # neuron_numbers = np.geomspace(1, 1000, num=10, dtype=int)
        # neuron_numbers = np.unique(np.concatenate(([1], neuron_numbers)))  # Ensure 1 is included and remove duplicates
        # seed_numbers = np.linspace(10, 1, num=len(neuron_numbers), dtype=int)
        # for i, n_neuron in enumerate(neuron_numbers):
            # for seed in range(1, seed_numbers[i]+1):
            
        # stim_numbers = np.geomspace(500, 30000, num=10, dtype=int)
        # stim_numbers = np.unique(stim_numbers)  # Remove duplicates that might occur due to rounding
        # for n_stim_train in stim_numbers:

        prefix = f'fullmodel_{mouse_names[mouse_id]}_{nlayers}_{nconv1}_{nconv2}_seed{seed}'
        bsub_cmd = f'bsub -n 2 -q gpu_{gpu} -gpu "num=1"  -J {prefix} -o {output_save_path}/{prefix}.out -e {output_save_path}/{prefix}.err "bash fullmodel_mouse_script.sh {gpu} {nlayers} {nconv1} {nconv2} {seed} {n_neuron} {n_stim_train} {weight_decay_core} {mouse_id} {lr} {l2_readout} {area} {pretrain_mouse_id}"'
        print(bsub_cmd)
        os.system(bsub_cmd)

if __name__=='__main__':
    main()
    