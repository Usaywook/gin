# This is repository for accepted paper to R-AL 2023 and will present to ICRA 2023.
Title of paper is [GIN: Graph-based Interaction-aware Constraint Policy Optimization for Autonomous Driving](https://ieeexplore.ieee.org/abstract/document/9976203)


## Prerequisite

### Installing CARLA physic engine
version 0.9.11
https://github.com/carla-simulator/carla/releases/tag/0.9.11/
download CARLA_0.9.11.tar.gz file
```
unzip CARLA_0.9.11.tar.gz
./CarlaUE.sh 
```

### Installing Conda virtual environment
change environment.yml file in line 239
{$user} to Hostname

```
conda env create --file environment.yml 
conda activate gin
```

## To train agents:
set RootPATH to user repository path

```
bash train.sh
```

## To test agents:
set RootPATH to user repository path
```
bash test.sh
```

## To get data:
set RootPATH to user repository path
set seed to run leaned agent
```
bash demo.sh
```
