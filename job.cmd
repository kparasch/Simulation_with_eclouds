#!/bin/bash

#BSUB -J tri_0011
#BSUB -o %J.out
#BSUB -e %J.err
#BSUB -N
#BSUB -B
#BSUB -q spacecharge
#B -a openmpi
#BSUB -n 4
#BSUB -R span[ptile=4]

#export PYTHONPATH=/home/giadarol/Desktop/PyPARIS_multibunch_development:$PYTHONPATH
rm simulation_status.sta 

#python /home/kparasch/Builds/PyPARIS/multiprocexec.py -n 4 sim_class=Simulation_with_eclouds.Simulation
python /home/kparasch/Builds/PyPARIS/serialexec.py sim_class=Simulation_with_eclouds.Simulation
