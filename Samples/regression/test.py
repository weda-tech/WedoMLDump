import sys

sys.path.append("./example/torchMLP")
from example.torchMLP.model import learningModel
from example.torchMLP.predict import predictor

# from predict import predictor
from wedaLib.reg import regTrainer
import json
import pandas as pd

## 처음부터 purposeType 분리해서 from wedaml import REG식으로 불러오기?

df = pd.read_csv("/Users/dmshin/gw/test/boston.csv")
target = "label"
trainDf = df[:-30]
testDf = df[-30:]

# hpo 할때 꼭 필요한 파라미터
# 파라미터, 범위, q, defaultValue
# hyperParamScheme = {
#   "gamma": {"min": 0, "max": 10, "type":"float", "q":0.05, "defaultValue": 1.3},
#   "n_d": {"min": 1, "max": 10, "type":"int", "defaultValue": 8}
# }

print("1")
model = regTrainer.train(df=trainDf, lm=learningModel, target=target)
# model = wml.train(trainDf, learningModel, target, graph=True)
# model = wml.train(trainDf, learningModel, target, xai=True)
# model = wml.train(trainDf, learningModel, target, graph=True, graphList=["regPlot"])

# print(model)


output2 = regTrainer.predict(predictor, model["result"]["saveMdlPath"], testDf, target)
print(output2)
