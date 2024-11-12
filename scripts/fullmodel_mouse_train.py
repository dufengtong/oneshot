import os
import numpy as np

mouse_names = ['L1_A5', 'L1_A1',  'FX9', 'FX10', 'FX8', 'FX20']
NNs = [6636,6055,3575,4792,5804,2746]

def main():
    mouse_id = 2
    for mouse_id in range(6):
        nconv1 = 192
        nconv2 = 192
        seed = 1
        nlayers = 1
        n_max_neurons = NNs[mouse_id] # Total number of neurons
        # n_max_stims = 4640 # Total number of unique train stimuli
        n_neuron = -1 # Number of neurons to sample
        n_stim_train = -1
        weight_decay_core = 0.1

        output_save_path = f'outputs/fullmodel/{mouse_names[mouse_id]}'
        if not os.path.exists(output_save_path):
            os.makedirs(output_save_path)

        # nconv1_list = [2,4,8,16,32,64, 128, 192, 256, 320, 384, 448]
        # nconv2_list = [2,4,8,16,32,64, 128, 192, 256, 320, 384, 448]

        # for nconv1 in nconv1_list:
        #     for nconv2 in nconv2_list:
        
        # for nlayers in range(1,5):

        # Generate lists of neuron numbers and seed numbers using logarithmic spacing
        # neuron_numbers = np.geomspace(1, 1000, num=10, dtype=int)
        # neuron_numbers = np.unique(np.concatenate(([1], neuron_numbers)))  # Ensure 1 is included and remove duplicates
        # seed_numbers = np.linspace(10, 1, num=len(neuron_numbers), dtype=int)
        # for i, n_neuron in enumerate(neuron_numbers):
            # for seed in range(1, seed_numbers[i]+1):
            
        # stim_numbers = np.geomspace(500, 30000, num=10, dtype=int)
        # stim_numbers = np.unique(stim_numbers)  # Remove duplicates that might occur due to rounding
        # for n_stim_train in stim_numbers:
        prefix = f'fullmodel_{mouse_names[mouse_id]}_{nlayers}_{nconv1}_{nconv2}_seed{seed}_downsample_2'
        bsub_cmd = f'bsub -n 2 -q gpu_t4 -gpu "num=1"  -J {prefix} -o {output_save_path}/{prefix}.out -e {output_save_path}/{prefix}.err "bash fullmodel_mouse_script.sh {nlayers} {nconv1} {nconv2} {seed} {n_neuron} {n_stim_train} {weight_decay_core} {mouse_id}"'
        print(bsub_cmd)
        os.system(bsub_cmd)

if __name__=='__main__':
    main()
    