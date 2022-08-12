#!/usr/bin/env bash
RootPATH=
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib
CARLAPATH=$RootPATH/carla2gym/carla/PythonAPI/dist/carla-0.9.11-py3.7-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:$CARLAPATH

############################### Test #######################################################################
N_episodes=100
seeds=(1 2 3 4 5)
for seed in ${seeds[@]}
do
    python main.py --validate_episodes=${N_episodes} --seed ${seed} --mode test --use_tracking --algo sac sac replay    

    python main.py --validate_episodes=${N_episodes} --seed ${seed} --mode test --use_tracking --algo wcsac wcsac replay

    python main.py --validate_episodes=${N_episodes} --seed ${seed} --mode test --use_graph --algo gin wcsac replay gin 
done

############################################################################################################

############################### Prediction Test ############################################################
seeds=(1 2 3 4 5)
for seed in ${seeds[@]}
do
    python prediction.py --mode test --study v-gru --seed ${seed} --algo grip grip

    python prediction.py --mode test --seed ${seed} --algo grip grip

    python prediction.py --mode test --use_graph --seed ${seed} --algo gin wcsac replay gin
done
############################################################################################################
