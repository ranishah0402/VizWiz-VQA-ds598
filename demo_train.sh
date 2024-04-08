#!/bin/bash -l

# Set SCC project
#$ -P ds598

module load miniconda
module load academic-ml/spring-2024

conda activate spring-2024-pyt

# Change this path to point to your project directory
export PYTHONPATH="/usr4/ds598/ranishah/VizWiz-VQA-ds598/API (1)/PythonHelperTools/vqaDemo.py:$PYTHONPATH"

python API/PythonHelperTools/vqaDemo.py

### The command below is used to submit the job to the cluster
### qsub -pe omp 4 -P ds598 -l gpus=1 demo_train.sh