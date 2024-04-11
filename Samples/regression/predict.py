from wedaLib.reg import regDecorator
import torch
import numpy as np


class predictor:

    @regDecorator.predict
    def runPredict(model, xTest):
        X_test_tensor = torch.tensor(xTest.values, dtype=torch.float32)
        model.eval()  # 평가 모드로 전환
        with torch.no_grad():  # 그라디언트 계산 비활성화
            yPred = model(X_test_tensor).numpy()
        print("===============")
        print(yPred)
        print("===============")
        return yPred

        # xTest = np.array(xTest)
        # y_pred = model.predict(xTest)

        # return y_pred
