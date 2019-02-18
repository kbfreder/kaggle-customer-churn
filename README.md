## Kaggle Customer Churn competition:

https://www.kaggle.com/c/kkbox-churn-prediction-challenge



### Data:

Data are available for download from the above link (they are too large to upload onto GitHub). Note they are compressed in .7z format, which on Mac/OS X requires specialty software to open. I have them saved in a folder named 'Data' on my local machine, and my code reflects this (i.e. `pd.read_csv('Data/<filename>')`. Another option is to start a kernel on Kaggle, in which case the datafile location becomes `pd.read_csv('../input/<filename')`.  The datafiles are versioned on Kaggle; I downloaded them in February 2019.

### Jupyter notebooks

Jupyter notebooks are written to demonstrate the decisions that went into data cleaning & feature selection. 

- 01-DataClean.ipynb: Exploration, aggregation, and dropping of data
- 02-FeatureSelection.ipynb: A bit more EDA, engineering & selection of features to include in model; testing of several Classification algorithms
- *03-TuneEvalModel: Tuning of CatBoost hyperparameters; evaluation again test (hold-out) set

### Python scripts

These can be run stand-alone to generate the output (cleaned data, preprocessors, fit models), given that the data has been downloaded (see Data section above). 

- data.py: cleans data
- *model.py: establishes feature set, fits data to CatBoost model, using tuned parameters established in Juptyer Notebook *03-TuneEvalModel.ipynb



**Note**: * indicates that file is incomplete / a "work in progress"