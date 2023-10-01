#!/bin/bash
#SBATCH --job-name=case-study-base-model
#SBATCH --output=case-study-base-model-%j.out
#SBATCH --error=case-study-base-model-%j.error
#SBATCH --chdir /home/tdewildt
#SBATCH --export=ALL
#SBATCH --get-user-env=L
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=00:30:00

# Setup modules
module purge
module load 2022
module load aiohttp/3.8.3-GCCcore-11.3.0
module load matplotlib/3.5.2-foss-2022a
module load PyTorch/1.12.0-foss-2022a-CUDA-11.7.0
module load SciPy-bundle/2022.05-foss-2022a
module load TensorFlow/2.11.0-foss-2022a-CUDA-11.7.0
module load torchvision/0.13.1-foss-2022a-CUDA-11.7.0

# Log versions
echo "PYTHON_VERSION = $(python --version)"
echo "PIP_VERSION = $(pip --version)"

# Setup environment
if [ ! -f $HOME/.env ]
then
  export $(cat $HOME/.env | xargs)
fi
export PYTHONPATH=${HOME}/master-thesis/src

# Setup dependencies
pip install -q \
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
python -m experiments.case_study_base_model
