
source ~/Desktop/add_anaconda.sh
eval "$(conda shell.bash hook)"
source ~/.bashrc

conda activate torchenv_"$1"

python fullmodel_mouse.py --nlayers "$2" --nconv1 "$3" --nconv2 "$4" --seed "$5" --n_neurons "$6" --n_stims "$7" --weight_decay_core "$8" --mouse_id "$9" --lr "${10}" --l2_readout "${11}" --area "${12}" 