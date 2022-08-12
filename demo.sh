#!/usr/bin/env bash
RootPATH=
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib
CARLAPATH=$RootPATH/carla2gym/carla/PythonAPI/dist/carla-0.9.11-py3.7-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:$CARLAPATH

seed=
############################## get demonstration ##########################################################
python demo.py --mode demo --validate_episodes 200 --use_graph --use_tracking --algo grip grip
python autopilot.py --mode demo --validate_episodes 20 --use_graph --seed ${seed}  --algo gin wcsac replay gin
###########################################################################################################
