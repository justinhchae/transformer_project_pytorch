# transformer_project_pytorch
 A starter template to get going with a hugging face transformer project with pytorch.

## Start Here: Environment Setup
* Create a conda environment from yml of base environment
  ```bash
  # replace transformer_project with your project name
  conda env create -n transformer_project --file=environment.yml
  ```

## Environment and Preliminaries from Scratch
* Conda Set Channel Strict conda-forge. Do this from base env prior to creating envs.
  ```bash
  # ensures a predictable selection of dependency management
  conda config --set channel_priority strict
  ```
* A new conda environment with python 3.8, or change to whatever version
  ```bash
  conda create -n transformer_project python=3.8
  ```
* Activate Conda Environment:
  ```bash
  conda activate transformer_project
  ```
* PyTorch conda install with Conda per [the docs](https://pytorch.org/get-started/locally/#start-locally)
  ```bash
  # This template made from macOS
  conda install pytorch torchvision torchaudio -c pytorch
  ```
* Transformers install with conda per [the docs](https://huggingface.co/transformers/installation.html)
  ```bash
  conda install -c huggingface transformers
  # verify with the following
  python -c "from transformers import pipeline; print(pipeline('sentiment-analysis')('we love you'))"
  ```
## General Resources Mostly Based on [Conda](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)
* Feature Resolver sometimes needed based on pip/conda/env
  ```bash
  # example: pip install <the package> --use-feature=2020-resolver
  pip install texthero --use-feature=2020-resolver
  ```
* Upgrade Pip:
  ```bash
  pip3 install --upgrade pip
  ```
* List Conda Environments:
  ```bash
  conda env list
  ```
* Remove/Delete Environment:
  ```bash
  conda remove -n transformer_project --all
  ```
* Display History of Revisions:
  ```bash
  conda list --revisions
  ```  
* Export Environment:
  ```bash
  conda env export > environment.yml
  ```  
* Update Conda:
  ```bash
  conda update -n base -c defaults conda
  ```
* Git Things:
  ```bash
  # remove a folder dir from git tracking
  git rm -r --cached .idea/
  # remove the DS store file but not the actual file
  git rm --cached .DS_Store
  # reset the cache
  git rm -r --cached .
  ```
