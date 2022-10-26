# PTR Implementation
This repository contains the implementation of the PTR algorithm described in the paper: [Pre-Training for Robots: Leveraging Diverse Multitask Data via Offline Reinforcement Learning](https://arxiv.org/abs/2210.05178).


## Code Structure
The code is structured as follows:
- `data_preprocessing/generate_numpy.py`: contains the code to preprocess the data.
- `jaxrl2/agents/cql_encoder_sep_parallel`: contains our parallelized implementation of the PTR algorithm. This code is builds on the ideas introduced in the [JAX TPU Colab](https://colab.research.google.com/github/google/jax/blob/master/docs/notebooks/quickstart.ipynb#scrollTo=5rmpybwysXNw). Note the parallelization environment is adaptive and does work with single GPU/CPU as well.
- `jaxrl2/utils`: contains the code for the environment and dataset wrappers.
- `examples/configs`: contains the config files for the dataset
- `examples/scripts`: contains the script(s) to run the experiment

## Sample Run Command
```bash
    XLA_PYTHON_CLIENT_PREALLOCATE=false python3 examples/launch_train_real_cql.py 
    --prefix resnet34_all_pp_data \
    --cql_alpha 10 \
    --encoder_type $encoder_type \
    --algorithm cql_encodersep_parallel \
    --dataset $data \ 
    --target_dataset $tdata \
    --batch_size 64 \
    --wandb_project $proj_name \
    --multi_viewpoint 0 \
    --add_prev_actions 0 \
    --policy_encoder_type same \
    --target_mixing_ratio 0.9 \
    --use_action_sep 1 \
    --use_basis_projection 0 \
    --discount 0.96 \
    --max_q_backup 1 \
    --num_final_reward_steps 3 \
    --term_as_rew 1 \
    --encoder_norm group \
    --use_spatial_learned_embeddings 1  \
    --target_entropy_factor 1.0 \
    --use_pixel_sep $pixel \
    --min_q_version 3 \
    --q_dropout_rate $drop \ 
    --offline_finetuning_start 160000
```


## Installation
### Directly from source
You can install the package by running the following command:
```bash
pip install --upgrade pip
python setup.py develop

pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

See instructions for other versions of CUDA [here](https://github.com/google/jax#pip-installation-gpu-cuda).

### Conda Environment Setup

Additionally, you can create a conda environment for a TPU machine with the required dependencies:


```bash
conda env create -f jax_tpu.yml
conda activate jax_tpu
```

The `jax_tpu.yml` file is located in the root directory of this repository.

## Public Datasets
You can find the datasets that were used for this paper [here](https://sites.google.com/view/ptr-robotlearning).

## Acknowledgements
Our repostiory is based off of the [JAX RL2](https://github.com/ikostrikov/jaxrl2) repository. We thank the authors for making their code public. We utilized on a earlier private version of the repository for our experiments. We have made the necessary changes to make it compatible with the latest version of the repository.