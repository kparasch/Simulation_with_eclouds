#!/bin/bash

#BSUB -J inoct_0014
#BSUB -o %J.out
#BSUB -e %J.err
#BSUB -N
#BSUB -B
#BSUB -q hpc_inf
#B -a openmpi
#BSUB -n 8
#BSUB -R span[ptile=8]

source setup_env_cnaf

CURRDIR=/home/HPC/aromano/sim_workspace_cnaf/033_LHC_instab_dip_and_quad_edens_scan_6p5TeV_octupoles/simulations/edens_10.00e11_ecdipON_ecquadON
cd $CURRDIR
pwd

stdbuf -oL python ../../../PyPARIS/multiprocexec.py -n 8 sim_class=Simulation_with_eclouds.Simulation >> opic.txt 2>> epic.txt
