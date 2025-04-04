# ISOMERIC (dISentangling cOpy nuMber dEpendent vaRiation In transCriptomes)

Copy number aberrations affect single-cell tumour expression profiles, but it is difficult to separate this from intrinsic transcriptomic variation. We introduce ISOMERIC, a deep learning framework that disentangles copy number-dependent and independent representations from scRNA-seq across multiple cancer types. ISOMERIC reveals distinct associations between these variations and clinical subtypes, links these variations to tumour microenvironment phenotypes, and establishes a principled approach for understanding genetic and intrinsic influences on cancer transcriptomes.

# Dependencies

Minimal dependencies for model training are:
- python 3.9
- torch 2.2.1
- numpy
- scipy 1.9.3
- scikit-learn 1.2.0
- scanpy 1.9.3
- anndata 0.9.1
- wandb

# Installation

Ensure Python 3.9 and the virtualenv management tool `pipenv` are available on your system. Clone the repo, navigate to the root, and run `pipenv install`. This will set up the virtual environment. Activate the environment with `pipenv shell`.

# Quickstart

Inputs to ISOMERIC include a scRNA-seq dataset filtered to tumour cells and a paired `csv` file holding copy number calls per chromosomal band for each cell.
A dataset should be registered in `train_model.set_dataset_params()` to make it accessible by name as a command line argument. Along with the `.h5ad` scRNA-seq file and `.csv` copy number calls file, the `.obs` column in the `.h5ad` file holding sample ID, the patient IDs to be used for the validation set, and the Weights & Biases project name will be specified in the function.

ISOMERIC can be trained with the following call from the command line inside the activated virtual environment:
```
python train_model.py --cohort peng --label_level band --epochs 2
```
The above will train ISOMERIC on the PDAC cohort (data pre-processed from Peng et al. (2019)) for 2 epochs. Other command line arguments control various hyperparameters and can be viewed with `python train_model.py -h`. The code supports logging with Weights & Biases and will track training in the specified project.
