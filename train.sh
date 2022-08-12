#!/usr/bin/env bash
RootPATH=
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib
CARLAPATH=$RootPATH/carla2gym/carla/PythonAPI/dist/carla-0.9.11-py3.7-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:$CARLAPATH

############################### Train ######################################################################
seeds=(1 2 3 4 5)
for seed in ${seeds[@]}
do
    python main.py --seed ${seed} \
    --use_tracking \
    --algo sac sac replay

    python main.py --seed ${seed} \
    --use_tracking \
    --algo wcsac wcsac replay

    python main.py --seed ${seed} \
    --use_graph --use_tp_graph \
    --algo gin wcsac replay gin
done
############################################################################################################

############################### Prediction Train ###########################################################
seeds=(1 2 3 4 5)
for seed in ${seeds[@]}
do
    python pretrain.py --mode pretrain --study v-gru --seed ${seed} --algo grip --num_epochs 300 grip

    python pretrain.py --mode pretrain --seed ${seed} --algo grip --num_epochs 300 grip
done
############################################################################################################
