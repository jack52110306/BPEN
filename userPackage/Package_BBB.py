from userPackage.Package_Encode import EncodeAllFeatures
from userPackage.LoadDataset import LoadDataset
import pandas as pd
from MLProcess.PycaretWrapper import PycaretWrapper
from MLProcess.Predict import Predict

class BBB_Predict:

    def __init__(self, model_use, pathDict):
        self.model_use = model_use
        self.pathDict = pathDict
        self.modelNameList = ['rbfsvm','lightgbm', 'xgboost', 'dt', 'et']
        self.dataList = []
        self.predVectorDf = None
        self.probVectorDf = None
        self.predVectorListIndp = None
        self.probVectorListIndp = None
        if self.model_use == '1':
            self.featureNum = 110
            self.featureTypeDictjson = 'BBB_featureTypeDict.json'
            self.nmlzPkl = 'BBB_standardScaler.pkl'
            self.featureRankCsv = '../Data/mlData/DS1/featureRank.csv'
        elif self.model_use == '2':
            self.featureNum = 270
            self.featureTypeDictjson = 'BBB_featureTypeDict.json'
            self.nmlzPkl = 'BBB_standardScaler.pkl'
            self.featureRankCsv = '../Data/mlData/DS2/featureRank.csv'
        elif self.model_use == '3':
            self.featureNum = 290
            self.featureTypeDictjson = 'BBB_featureTypeDict.json'
            self.nmlzPkl = 'BBB_standardScaler.pkl'
            self.featureRankCsv = '../Data/mlData/DS3/featureRank.csv'
        elif self.model_use == '4':
            self.featureNum = 170
            self.featureTypeDictjson = 'BBB_featureTypeDict.json'
            self.nmlzPkl = 'BBB_standardScaler.pkl'
            self.featureRankCsv = '../Data/mlData/DS4/featureRank.csv'
        else:
            raise NameError('model_use should input 1 or 2 or3 or 4, 1 = DS1, 2 = DS2,3 = DS3, 4 = DS4')

    def loadData(self, inputDataList):
        ldObj = LoadDataset()
        for inputData in inputDataList:
            testSeqDict = ldObj.readFasta(inputData)
            self.dataList.append(testSeqDict)

    def featureEncode(self):
        encodeObj = EncodeAllFeatures()
        encodeObj.dataEncodeSetup(loadJsonPath=f'{self.pathDict["paramPath"]}/{self.featureTypeDictjson}',
                                  b_loadJson=True)

        encoded_data = encodeObj.dataEncodeOutPut(dataDict={0: self.dataList[0], 1: self.dataList[1], -1: None})

        testDf = encodeObj.dataNormalization(encodeIndpDf=encoded_data,
                                             loadNmlzScalerPklPath=f'{self.pathDict["paramPath"]}/{self.nmlzPkl}',
                                             b_loadPkl=True)

        featureDf = pd.read_csv(self.featureRankCsv)

        # 【神級修正：擷取前 N 名】
        # 依照 self.featureNum 的數量 (例如 110)，只取排行榜最前面的特徵！
        top_n = int(self.featureNum)
        featureList = featureDf['feature name'].head(top_n).to_list()

        # 使用 reindex 自動對齊並補 0
        testDf1 = testDf.reindex(columns=featureList, fill_value=0)

        testDf1.to_csv(f'{self.pathDict["saveCsvPath"]}/test_F{self.featureNum}.csv')
    '''def featureEncode(self):
        encodeObj = EncodeAllFeatures()
        encodeObj.dataEncodeSetup(loadJsonPath=f'{self.pathDict["paramPath"]}/{self.featureTypeDictjson}', b_loadJson=True)
        encoded_data = encodeObj.dataEncodeOutPut(dataDict={0: self.dataList[0], 1: self.dataList[1], -1: None})
        testDf = encodeObj.dataNormalization(encodeIndpDf=encoded_data,
                                             loadNmlzScalerPklPath=f'{self.pathDict["paramPath"]}/{self.nmlzPkl}',
                                             b_loadPkl=True)
        featureDf = pd.read_csv(self.featureRankCsv)
        featureList = featureDf['feature name'].to_list()
        testDf1 = testDf[featureList]
        testDf1.to_csv(f'{self.pathDict["saveCsvPath"]}/test_F{self.featureNum}.csv)'''

    def doPredict(self):
        pycObj = PycaretWrapper()
        modelList = pycObj.doLoadModel(path=self.pathDict['modelPath'], fileNameList=self.modelNameList)
        dataTestDf = pd.read_csv(f'{self.pathDict["saveCsvPath"]}/test_F{self.featureNum}.csv', index_col=[0])
        predObjIndp = Predict(dataX=dataTestDf, modelList=modelList)
        self.predVectorListIndp, self.probVectorListIndp = predObjIndp.doPredict()
        self.predVectorDf = pd.DataFrame(self.predVectorListIndp, index=self.modelNameList, columns=dataTestDf.index).T
        self.probVectorDf = pd.DataFrame(self.probVectorListIndp, index=self.modelNameList, columns=dataTestDf.index).T
        self.predVectorDf.to_csv(f'{self.pathDict["outputPath"]}/binary_vector.csv')
        self.probVectorDf.to_csv(f'{self.pathDict["outputPath"]}/probability_vector.csv')

