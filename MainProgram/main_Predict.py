from userPackage.Package_BBB import BBB_Predict

# If you want to use different model, you can change dataset
dataset = 'DS4'

if dataset == 'DS1':
    model_use = '1'
elif dataset == 'DS2':
    model_use = '2'
elif dataset == 'DS3':
    model_use = '3'
elif dataset == 'DS4':
    model_use = '4'

# Path setting
pathDict = {'paramPath': f'../Data/param/{dataset}/',  # This path should have featureTypeDict.pkl and robust.pkl
            'saveCsvPath': '../Data/mlData/new_data/',  # Your encoded data will save in this path
            'modelPath': f'../Data/finalModel/{dataset}/',  # This path should have rbfsvm, lightgbm, gbc models. ex: gbc_final.pkl
            'outputPath': f'../Data/output/'}  # Your prediction will save in this path

# Input your FASTA file, the example file can find in data/mlData/Hmp1/test_neg.FASTA
inputPathList = [f'../Data/mlData/{dataset}/test_neg.FASTA', f'../Data/mlData/{dataset}/test_pos.FASTA']

encapObj = BBB_Predict(model_use=model_use, pathDict=pathDict)
encapObj.loadData(inputDataList=inputPathList)
encapObj.featureEncode()
encapObj.doPredict()
