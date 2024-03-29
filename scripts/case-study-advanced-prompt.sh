#!/bin/bash
#SBATCH --job-name=case-study-advanced-prompt
#SBATCH --output=case-study-advanced-prompt-%j.out
#SBATCH --error=case-study-advanced-prompt-%j.error
#SBATCH --chdir /home/tdewildt/master-thesis
#SBATCH --export=ALL
#SBATCH --get-user-env=L
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=t.l.dewildt@tilburguniversity.edu

# Setup modules
if [ ! -z "${SLURM_JOB_ID}" ]; then
    module purge
    module load 2022
    module load aiohttp/3.8.3-GCCcore-11.3.0
    module load matplotlib/3.5.2-foss-2022a
    module load PyTorch/1.12.0-foss-2022a-CUDA-11.7.0
    module load SciPy-bundle/2022.05-foss-2022a
    module load TensorFlow/2.11.0-foss-2022a-CUDA-11.7.0
    module load torchvision/0.13.1-foss-2022a-CUDA-11.7.0
fi

# Log versions
echo "PYTHON_VERSION = $(python3.10 --version)"
echo "PIP_VERSION = $(pip3.10 --version)"

# Setup environment
if [ -f .env ]; then
    export $(cat .env | xargs)
fi
export PYTHONPATH=src

# Setup dependencies
pip3.10 install -q \
    accelerate \
    anthropic \
    bert-score \
    bitsandbytes \
    git+https://github.com/google-research/bleurt.git \
    datasets \
    google-cloud-aiplatform \
    langchain \
    nltk \
    openai \
    peft \
    protobuf \
    rapidfuzz \
    rouge-score \
    scipy \
    sentencepiece \
    tiktoken \
    torch \
    transformers \
    trl \
    xformers

# Run experiment
python3.10 -m experiments.case_study_advanced_prompt
