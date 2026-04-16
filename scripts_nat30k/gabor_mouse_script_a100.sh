
source ~/Desktop/add_anaconda.sh
eval "$(conda shell.bash hook)"
source ~/.bashrc

conda activate torchenv_a100

python gabor_mouse.py --mouse_id "$1" --seed "$2" --n_neurons "$3" --n_stims "$4"


