#!/bin/bash


#$ -M jboumalh@nd.edu   # Email address for job notification
#$ -m abe                # Send mail when job begins, ends and aborts
#$ -pe smp 1             # Specify parallel environment and legal core size
#$ -q gpu                # Run on the GPU cluster
#$ -l gpu_card=1         # Run on 1 GPU card
#$ -N Salamander_RB      # Specify job name

module load gromacs  # Required modules

export OMP_NUM_THREADS=$NSLOTS
gmx mdrun -ntomp $OMP_NUM_THREADS -nb gpu -pin on -v -s input.tpr # Run with 16 MPI tasks and 1 GPU devices

module load python/3.8.10  

python3 /home/minirolab/Desktop/Salamander/ROS-WorkSpace/src/salamander_development/scripts/James_Exp1.py