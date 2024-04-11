from wedaLib.reg import regDecorator
import torch
import torch.nn as nn
import torch.optim as optim


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(13, 64)  # 입력 차원: 13, 은닉층 크기: 64
        self.fc2 = nn.Linear(64, 64)  # 첫 번째 은닉층 크기: 64, 두 번째 은닉층 크기: 64
        self.fc3 = nn.Linear(64, 1)  # 출력 차원: 1

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # 첫 번째 은닉층에 ReLU 활성화 함수 적용
        x = torch.relu(self.fc2(x))  # 두 번째 은닉층에 ReLU 활성화 함수 적용
        x = self.fc3(x)  # 출력층
        return x


class learningModel:
    @regDecorator.wedaLearningModel
    def __init__(self, *args, **kwargs):
        print("gogogo")
        pass

    def createModel(self):
        self.model = MLP()
        return self.model

    # return은 model + @
    @regDecorator.fit
    def fit(self, X: list, y: list):

        X_train_tensor = torch.tensor(X, dtype=torch.float32)
        y_train_tensor = torch.tensor(y, dtype=torch.float32)

        epochs = 1000
        criterion = nn.MSELoss()  # 평균 제곱 오차
        optimizer = optim.Adam(
            self.model.parameters(), lr=0.001
        )  # Adam 옵티마이저 사용

        for epoch in range(epochs):
            # Forward pass
            outputs = self.model(X_train_tensor)
            loss = criterion(outputs.squeeze(), y_train_tensor)

            # Backward pass 및 경사도 업데이트
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

        # y = y.reshape(-1, 1)

        # self.model.fit(X, y)
        return self.model

    # return은 yPred + @
    @regDecorator.validation
    def validation(self, xTest, yTest, model):
        X_test_tensor = torch.tensor(xTest, dtype=torch.float32)
        self.model.eval()  # 평가 모드로 전환
        with torch.no_grad():  # 그라디언트 계산 비활성화
            yPred = self.model(X_test_tensor).numpy()

        return yPred
