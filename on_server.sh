#!/bin/bash
# Instructions SBATCH always at the beginning of the script!
# The job partition (maximum elapsed time of the job).

#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G

# The name of the job.
#SBATCH -J current_log

# The number of GPU cards requested.
##SBATCH --gres=gpu:ampere:1

# Email notifications (e.g. the beginning and the end of the job).
##SBATCH --mail-user=ganglin.tian@lmd.ipsl.fr
#SBATCH --mail-user=baiyunbuaa@gmail.com
## SBATCH --mail-user=haofengqingyang@gmail.com
# SBATCH --mail-type=all
## SBATCH --mail-type=INVALID_DEPEND,END,FAIL,REQUEUE,STAGE_OUT

# The path of the job log files.
# The error and the output logs can be merged into the same file.
# %j implements a job counter.
#SBATCH --error=server_log/current_log
#SBATCH --output=server_log/current_log

# Overtakes the system memory limits.
ulimit -s unlimited

clear
python -u benchmark_models.py

##############################################################
# 把储存在/home/tganglin/yun/large_files中的大文件, 以快捷键的方式soft link到你的目标文件夹
# ln -s /home/tganglin/yun/large_files/text_features.csv /home/tganglin/yun/02-ProbabilisticFore/ResultsOct2023/TextFeats_process/

##############################################################
# 使用说明
##############################################################
# 运行你的scripts
# sbatch on_server.sh

# 停止你的job
# scancel 

# 查看当前你的任务是在排队还是在运行
# watch -n 1 squeue -u tganglin

# 查看当前输出
# tail -f server_log/current_log

# 查看JOB ID 这个任务的运行配置 
# scontrol show job 你的ID

# 或者
# sacct -u tganglin
# sacct -u tganglin -j 你的ID

# conda create -n yun_env python=3.9 -y
# pip install matplotlib
# pip install -U statsmodels
# conda install -c conda-forge matplotlib optuna pandas scikit-learn statsmodels -y

# 实时更新node状态
# watch -n 1 check-cluster