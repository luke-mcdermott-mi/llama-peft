# Directions
### Set up Environment
Create a virtual environment: `` python3 -m venv .env ``

Load the virtual environment: ``source .env/bin/activate``

Install dependencies: ``pip install -r requirements.txt``

Set HF token
``export HF_HOME=<directory>``
``export HUGGINGFACE_HUB_TOKEN=<token>``

### Simple LoRA finetuning run with Llama-2-7b on SciQ
``python sciq.py``

### Set up Accelerate for Multi-GPU training
Type ``accelerate config`` in terminal and configure your environment

### Run with Accelerate
To launch a run across multiple gpus, type ``accelerate launch sciq.py``
