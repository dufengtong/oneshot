import os
import numpy as np

mouse_names = ['L1_A5', 'L1_A1',  'FX9', 'FX10', 'FX8', 'FX20', 'FX40', 'FX41', 'FX43', 'FX42', 'FX41']
NNs = [6636,6055,3575,4792,5804,2746, 4261,0,0,6049,0]

def main():
    mouse_id = 5
    for mouse_id in [7,8,9]:
        seed = 1
        n_neuron = -1 # Number of neurons to sample
        n_stim_train = -1 # 5000 # -1

        
        output_save_path = f'outputs/gabormodel'
        if not os.path.exists(output_save_path):
            os.makedirs(output_save_path)

        prefix = f'gabormodel_{mouse_names[mouse_id]}_seed{seed}'
        bsub_cmd = f'bsub -n 2 -q gpu_a100 -gpu "num=1"  -J {prefix} -o {output_save_path}/{prefix}.out -e {output_save_path}/{prefix}.err "bash gabor_mouse_script_a100.sh {mouse_id} {seed} {n_neuron} {n_stim_train}"'
        print(bsub_cmd)
        os.system(bsub_cmd)

if __name__=='__main__':
    main()