
source ~/Desktop/add_anaconda.sh
eval "$(conda shell.bash hook)"
source ~/.bashrc

conda activate torchenv_a100

python minimodel_mouse.py --nlayers "$1" --nconv1 "$2" --nconv2 "$3" --seed "$4" --ineuron "$5" --n_stims "$6" --mouse_id "$7" --hs_readout "$8" --pretrain_mouse_id "$9"