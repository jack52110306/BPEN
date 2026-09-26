# BBB Project

BBB: A machine learning framework for prediction for Blood Brain Barrier penetrating peptide using diverse physicochemical and compositional features

## Description

This is the source code of BBB, a machine learning predictor for prediction of Blood Brain Barrier penetrating peptides. The first stage is the optimization of feature vector, and the second stage is the hyperparameter tuning of the machine learning models. The trained models are also included in this package, and can help in prediction on a given peptide data set.

## Installation

Requiremenets:

* Python = 3.8, pycaret[full] = 2.3.10

Packages

* Install required packages using `pip install -r requirements.txt`

## Usage

Modify main_predict.py for your data set in fasta format

* Input file

    * One or more files in Fasta format (described below)

* output file

    * binary_vector.csv -- The prediction output in binary format (1 for positive and 0 for negative)

      <img width="439" height="305" alt="image" src="https://github.com/user-attachments/assets/ccc2ed6c-49bc-4c75-8d37-a2d42e690f38" />

    * probability.csv -- The prediction probability estimate
   
      <img width="439" height="407" alt="image" src="https://github.com/user-attachments/assets/6d5bb7cf-e937-449e-9412-209d3b6263e6" />

When dataset = 'DS1', the program will use models trained on DS1, corresponding features and their normalization scaler to process data and perform prediction.

When dataset = 'DS2' , the program will use models trained on DS2, corresponding features and their normalization scaler to process data and perform prediction.

When dataset = 'DS3' , the program will use models trained on DS3, corresponding features and their normalization scaler to process data and perform prediction.

When dataset = 'DS4' , the program will use models trained on DS4, corresponding features and their normalization scaler to process data and perform prediction.

```python
# If you want to use different model, you can change dataset here
if dataset == 'DS1':
    model_use = '1'
elif dataset == 'DS2':
    model_use = '2'
elif dataset == 'DS3':
    model_use = '3'
elif dataset == 'DS4':
    model_use = '4'
```

```python
# Path setting
pathDict = {'paramPath': f'../Data/param/{dataset}/',  # This path should have featureTypeDict.pkl and robust.pkl
            'saveCsvPath': '../Data/mlData/new_data/',  # Your encoded data will save in this path
            'modelPath': f'../Data/finalModel/{dataset}/',  # This path should have rbfsvm, lightgbm, gbc models. ex: gbc_final.pkl
            'outputPath': f'../Data/output/'}  # Your prediction will save in this path
```            

Specify one or more fasta files in the 'inputPathList' parameter. Sequences from these fasta files will be concatenated for prediction, and prediction results will be written to the default output files, binary_vector.csv and probability.csv.

```python
# Input your FASTA file, the example file can find in data/mlData/Hmp1/test_neg.FASTA
inputPathList = [f'../Data/mlData/{dataset}/test_neg.FASTA', f'../Data/mlData/{dataset}/test_pos.FASTA']
```
 
Here is the code snippet in main_predict.py. We already set the parameters and the program is ready to be excecuted.

```python
encapObj = BBB_Predict(model_use=model_use, pathDict=pathDict)
encapObj.loadData(inputDataList=inputPathList)
encapObj.featureEncode()
encapObj.doPredict()
``` 
