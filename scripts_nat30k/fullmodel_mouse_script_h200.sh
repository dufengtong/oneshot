
source ~/Desktop/add_anaconda.sh
eval "$(conda shell.bash hook)"
source ~/.bashrc

conda activate torchenv_h200

python fullmodel_mouse.py --nlayers "$1" --nconv1 "$2" --nconv2 "$3" --seed "$4" --n_neurons "$5" --n_stims "$6" --weight_decay_core "$7" --mouse_id "$8" --conv1_ks "$9" --conv2_ks "${10}" --pretrain_mouse_id "${11}" --hs_readout "${12}"


