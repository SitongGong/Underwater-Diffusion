#!/usr/bin/env bash
source ~/miniconda3/bin/activate pytorch
conda env list
cd /18515601223/UnderwaterDiffusion

accelerate launch train_gligen_lora.py