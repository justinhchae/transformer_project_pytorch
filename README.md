# transformer_project_pytorch
 * A starter template to get going with a hugging face transformer project with pytorch.
 * Template Source: [https://github.com/justinhchae/transformer_project_pytorch](https://github.com/justinhchae/transformer_project_pytorch)
 
## Why use this template?
 * To experiment with various data structures, models, and methods that involve transformers and language modeling
 * Learn more about Hugging Face, Bert, and PyTorch
 * Leverage state of the art pretrained models from saved model check points for comparison testing locally 

## Environment Setup with Conda ENV
* Create a conda environment from yml of base environment
  ```bash
  # replace transformer_project with your project name
  conda env create --file environment.yml -n transformer_project
  ```
## Get Started
* Run a full pipeline of text classification scripts with bert and corpora through pytorch:
```bash
python3 main.py
```
* Or run this from [Google Colab](https://colab.research.google.com/drive/1ovTQih-iCt_0yeTOqw17gL0az8qyE-bF?usp=sharing)

## Glue/Jiant Resources:
* [NYU Jiant](https://github.com/nyu-mll/jiant)

## Hugging Face Bert PyTorchExamples/Resources:
* [https://github.com/huggingface/transformers](https://github.com/huggingface/transformers)
* [https://trishalaneeraj.github.io/2020-04-04/feature-based-approach-with-bert](https://trishalaneeraj.github.io/2020-04-04/feature-based-approach-with-bert)
* [https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html](https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html)
* [https://mccormickml.com/2019/07/22/BERT-fine-tuning/](https://mccormickml.com/2019/07/22/BERT-fine-tuning/)
* [https://trishalaneeraj.github.io/2020-04-04/feature-based-approach-with-bert](https://trishalaneeraj.github.io/2020-04-04/feature-based-approach-with-bert)
## Environment and Preliminaries from Scratch with Conda and Pip
* Conda Set Channel Strict conda-forge. Do this from base env prior to creating envs.
  ```bash
  # ensures a predictable selection of dependency management
  conda config --set channel_priority strict
  ```
* A new conda environment with python 3.7; seems to be required to support jiant environment
  ```bash
  conda create -n transformer_project python=3.7
  ```
* Activate Conda Environment:
  ```bash
  conda activate transformer_project
  ```
* Install all dependencies with pip
  ```bash
  pip3 install --no-cache-dir huggingface huggingface_hub transformers jiant torch torchvision torchaudio progressbar2 tqdm boto3 requests regex sentencepiece sacremoses pandas scikit-learn matplotlib
  ```
* Not Required if using previous pip installs - PyTorch conda install with Conda per [the docs](https://pytorch.org/get-started/locally/#start-locally)
  ```bash
  conda install pytorch torchvision torchaudio -c pytorch
  ```
* Not Required if using previous pip installs - Transformers install with conda per [the docs](https://huggingface.co/transformers/installation.html)
  ```bash
  conda install -c huggingface transformers
  # verify with the following
  python -c "from transformers import pipeline; print(pipeline('sentiment-analysis')('we love you'))"
  ```

## General Resources Mostly Based on [Conda](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)
* Feature Resolver sometimes needed based on pip/conda/env
  ```bash
  # becoming less necessary and not required in 2021, might be better to just upgrade pip
  pip install <package_name> --use-feature=2020-resolver
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
