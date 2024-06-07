## LATnet
LATnet is a robust prediction model for the subcellular localization of LncRNA and mRNA. It uses an improved Transformer for feature extraction and a dual-channel parallel structure for training.

### Getting Started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

Prerequisites
The project requires the following Python packages:
```
lightning_attn==0.0.4
numpy==1.22.4
numpy==1.22.0
pandas==1.3.1
scikit_learn==0.24.0
torch==1.11.0+cu113
rna-fm==0.1.2
```

You can install these packages using pip:  

`pip install -r requirements.txt` 


### Running the Project
1. Start by generating the embedding files for the lncRNA and mRNA training and test sets using `rna-fm.ipynb`. This will create the corresponding RNA directory in the dataset.
2. If you are working with lncRNA, run the training with the command `python main.py`.
3. If you are working with mRNA, use the command `python mRNA_main.py`.


### Data availability
The dataset directory contains the original RNA files for both the training and test sets.

### Project Structure
The project has the following structure:  
```
.
├── __pycache__
├── .gitignore
├── dataset
│   ├── lncRNA
│   │   ├── lncRNA_sublocation_TestSet.tsv
│   │   └── lncRNA_sublocation_TrainingSet.tsv
│   └── mRNA
│       ├── mRNA_sublocation_TestSet.tsv
│       └── mRNA_sublocation_TrainingSet.tsv
├── dataset.py
├── main.py
├── model
│   ├── __pycache__
│   ├── liner_attention.py
│   ├── model.py
│   └── mRNA_model.py
├── mRNA_dataset.py
├── mRNA_main.py
├── PolyLoss.py
├── Readme.md
├── requirements.txt
├── rna-fm.ipynb
├── test.py
├── train.py
└── utils.py
```

The project contains several main parts:  
* dataset/: This directory contains the training and testing datasets, which are stored in the lncRNA/ and mRNA/ subdirectories respectively.
* model/: This directory contains the model code for the project, including model.py and mRNA_model.py.
* main.py and mRNA_main.py: These two files are the main entry points of the project, used for training the lncRNA and mRNA models respectively.
* dataset.py and mRNA_dataset.py: These two files are used for processing the datasets.
* PolyLoss.py: This file defines the loss function.
* requirements.txt: This file lists the dependencies of the project.
* rna-fm.ipynb: This Jupyter notebook is used for generating embedding files.
* test.py and train.py: These two files contain the testing and training code.
* utils.py: This file contains some utility functions.