import os
import numpy as np

mouse_names = ['L1_A5', 'L1_A1',  'FX9', 'FX10', 'FX8', 
               'FX20', 'FX40', 'FX41', 'FX43', 'FX42', 
               'FX41', 'FX41', 'FX43']
NNs = [6636,6055,3575,4792,5804,
       2746, 4261,0,0,6049,
       5247,3491, 4180]
NNs_valid = [4242,2840,926,3040,2217,
             1239, 0,2068,1655,0,
             886,306,1681]

def main():
    mouse_id = 0
    for mouse_id in [7]:
        nconv1 = 16
        nconv2 = 64
        seed = 1
        nlayers = 2
        n_max_neurons = NNs_valid[mouse_id] # Total number of neurons
        n_stim_train = -1
        hs_readout = 0.03 # 0.03 # 0.1 for 5k
        ineurons = np.arange(n_max_neurons)
        pretrain_mouse_id = -100

        hs_list = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.5]

        # param search
        # np.random.seed(0)
        # ineurons = np.random.choice(ineurons, 10, replace=False)

        # param search sup figure
        np.random.seed(42)
        # ind_selected = np.random.choice(np.arange(np.sum(NNs_valid)), 10, replace=False)
        # ind_all = np.zeros(np.sum(NNs_valid), dtype=bool)
        # ind_all[ind_selected] = True
        # ineurons_all = []
        # nmouse = len(NNs_valid)
        # for i in range(nmouse):
        #     if i == 0:
        #         ineurons_all.append(np.where(ind_all[:NNs_valid[i]])[0])
        #     else:
        #         ineurons_all.append(np.where(ind_all[np.sum(NNs_valid[:i]):np.sum(NNs_valid[:i+1])])[0])
        # ineurons = ineurons_all[mouse_id]
        # ineurons = [737, 742]
        ineurons = np.random.choice(ineurons, 2, replace=False)

        # output_save_path = f'outputs/minimodel_param_search/{mouse_names[mouse_id]}'
        output_save_path = f'outputs/minimodel/{mouse_names[mouse_id]}'
        if not os.path.exists(output_save_path):
            os.makedirs(output_save_path)

        # reuse conv1
        # np.random.seed(42)
        # ineurons = np.random.choice(ineurons, 100, replace=False)
        # for pretrain_mouse_id in [5]:
        # for hs_readout in hs_list:
        for ineuron in ineurons:
            prefix = f'minimodel_{mouse_names[mouse_id]}_{nlayers}_{nconv1}_{nconv2}_seed{seed}_nn{ineuron}'
            bsub_cmd = f'bsub -n 2 -q gpu_a100 -gpu "num=1"  -J {prefix} -o {output_save_path}/{prefix}.out -e {output_save_path}/{prefix}.err "bash minimodel_mouse_script_a100.sh {nlayers} {nconv1} {nconv2} {seed} {ineuron} {n_stim_train} {mouse_id} {hs_readout} {pretrain_mouse_id}"'
            print(bsub_cmd)
            os.system(bsub_cmd)

if __name__=='__main__':
    main()
    