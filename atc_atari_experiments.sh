#!/bin/bash

#SBATCH --time=108:00:00
#SBATCH --mem=12G
#SBATCH --mail-user=miahi@ualberta.ca 
#SBATCH --mail-type=ALL
#SBATCH --output=atc_atari_experiments-%x-%j.out
#SBATCH --error=atc_atari_experiments-%j-%n-%a.err
#SBATCH --gres=gpu:v100l:1
#SBATCH --account=rrg-whitem

module load python/3.7

source $HOME/rlpyt/bin/activate

taskset -c 0,1,2,3,4,5,6,7,8,9,10,11 /home/miahi/rlpyt/bin/python rlpyt/ul/experiments/rl_with_ul/scripts/atari/train/atari_dqn_with_ul_serial.py 0slt_12cpu_1gpu_0hto_2skt /project/6010404/erfan/github/rlpyt/data/local/20211006/140220/atari_dqn_with_ul_schedule_1/Falsestpcnvgrd/100000.0rlminstepsul50000.0_cosineanneal/quadratic_3/breakout 0 scaled_ddqn_ul
