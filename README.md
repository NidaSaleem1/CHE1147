
This repository contains the structured workflow for our CHE1147 project, covering everything from initial data exploration to final model analysis.

---

## Project Structure

### Source Code
The code is organized into four main Python scripts (present in src folder), designed to be executed sequentially

| Script Name| Key Responsibilities |
| :--- | :--- |
| **`preprocess.py`** | Clean the raw data, handle missing values (imputation/dropping), encode categorical features, scale/normalize numerical features, and prepare the final training and testing datasets. |
| **`exploratory_data_analysis.py`** | Perform initial data inspection for visualization. |
| **`train_model.py`** | Define multiple model architectures, train and test models via cross validation |
| **`result_analysis.py`** | Analyze the models and reasons for behaviors observed during model training  |

### Data
The data folder contains the following three essential files:
| File name | Description |
| :--- | :--- |
| **`*data.csv`** | Contains the main data set |
| **`CHE1147_Amine.xlsx`** | |
| **`CHE1147_Supports.xlsx`** | |

---

## Getting Started

### Installation

1.  Clone this repository:
    ```bash
    git clone https://github.com/NidaSaleem1/CHE1147.git
    cd CHE1147
    ```
2.  Create and activate a new Conda environment (recommended):
    ```bash
    conda env create -f environment.yml
    conda activate demo
    ```

### Execution Order

The scripts are designed to be run in the following order:

1.  `python src\preprocess.py`
    
    After **preprocess.py** is run, **df_test.csv** and **df_train.csv** which are test and train sets respectively are stored in the data folder to be used by other stags. Sample of these files are present in the data folder.
2.  `python src\exploratory_data_analysis.py`
3.  `python src\model.py`
4.  `python src\result_analysis.py`
