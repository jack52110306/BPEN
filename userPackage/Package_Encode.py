from devPackage.PackageModelAmp import EncodeModelAmp
from devPackage.PackageiFeature import iFeature
from devPackage.PackagePFeature import PFeature
from devPackage.OVP import OVP
from devPackage.MotifBitVec import MotifBitVec
from devPackage.Normalization import Normalization
import pandas as pd
import pickle
import json


class EncodeAllFeatures:
    def __init__(self):
        self.indpDf = None
        self.loadFeatureDict = None

    def dataEncodeSetup(self, loadJSONPath):
        path = loadJSONPath
        with open(path, 'r') as json_file:
            self.loadFeatureDict = json.load(json_file)

    def dataEncodeOutput(self, dataList):
        encodedDfList = []
        for inputData in dataList:
            eifObj = iFeature(inputData, self.loadFeatureDict['iFeature'])
            epfObj = PFeature(inputData, self.loadFeatureDict['pFeature'])
            emaObj = EncodeModelAmp(inputData, self.loadFeatureDict['ampFeature'])  # windows 拉出去dict
            eovpObj = OVP(inputData, self.loadFeatureDict['ovpFeature'])
            a = eifObj.getOutputDf()
            b = epfObj.getOutputDf()
            c = emaObj.getOutputDf()
            d = eovpObj.getOutputDf()
            encodedDf = pd.concat([a, b], axis=1)
            encodedDf = pd.concat([encodedDf, c], axis=1)
            encodedDf = pd.concat([encodedDf, d], axis=1)
            encodedDfList.append(encodedDf)

        if len(encodedDfList) >= 2:
            indpDf = pd.concat(encodedDfList)
        else:
            indpDf = encodedDfList[0]

        self.indpDf = indpDf

    def dataNormalization(self, loadNmlzScalerPklPath='./data/'):
        nmlzObj = Normalization(testDf=self.indpDf)
        indpNmlzDf = nmlzObj.robustTest(loadNmlzParamsPklPath=loadNmlzScalerPklPath)
        self.indpDf = indpNmlzDf

        return self.indpDf
