#!/bin/sh
## check idle gpus with sinfo -o "%.10P %.5a %.10l %.6D %.6t %.20N %.10G"
## change partition and nodelist accordingly

## Job Name

## TODO CHANGE THIS
#SBATCH --job-name=Kannada2Indicgemma

## Partition and Priority

## TODO CHANGE THIS
#SBATCH --partition=gpuq           # GPU partition for jobs requiring GPUs

#SBATCH --qos=gpu                  # GPU quality of service (required for GPU jobs)

## Output and Error Files
#SBATCH --output=/scratch/%u/%x-%N-%j.out  # Save output log
#SBATCH --error=/scratch/%u/%x-%N-%j.err   # Save error log

## Resource Specifications
#SBATCH --gres=gpu:A100.80gb:1        # Request one A100 GPU with 8 GB memory
##SBATCH --mem-per-cpu=6000M 

#SBATCH --mem-per-cpu=4G
##SBATCH --mem=128GB                 # Memory allocation per task

## TODO CHANGE THIS
##SBATCH --time=1-00:00:00
#SBATCH --time=1-00:00:00          # Max time for the job (1 day)

## Email Notifications
#SBATCH --mail-type=END,FAIL       # Notify on job end and failure
#SBATCH --mail-user=mrinki@gmu.edu # Use your GMU email address for notifications

## Task and Node Specifications
#SBATCH --nodes=1                  # Run on a single node
#SBATCH --ntasks-per-node=4          # Use 4 CPU cores for auxiliary tasks

## TODO CHANGE THIS
##SBATCH --nodelist=gpu011 # If you want to run on a specific node

## Load Necessary Modules
module load gnu10
module load cuda/11.2.2-5f
##module load cuda/12.6.3
module load python                 # Ensure Python environment is loaded


## Activate Virtual Environment
source venv/bin/activate           # Activate your Python virtual environment

## Run the Dataset Generation Script
##python generateDataset_mt0xxl.py          # Replace with your script filename

python generateDataset_indicgemma.py 


