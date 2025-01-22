import torch
import torch.nn as nn


# 정상 데이터만을 학습 -> 정상 데이터의 제구성 오류(Reconstruction Error)를 최소화하도록 훈련

class Autoencoder(nn.Module):
    def __init__(self,input_dim, block_size):
        super(Autoencoder,self).__init__()
        self.input_dim = input_dim
        self.cov_source = nn.Parameter(torch.zeros(block_size, block_size), requires_grad=False)
        self.cov_target = nn.Parameter(torch.zeros(block_size, block_size), requires_grad=False)

        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim,128),
            nn.BatchNorm1d(128,momentum=0.01, eps=1e-03),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.BatchNorm1d(128,momentum=0.01, eps=1e-03),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.BatchNorm1d(128,momentum=0.01, eps=1e-03),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.BatchNorm1d(128,momentum=0.01, eps=1e-03),
            nn.ReLU(),
            nn.Linear(128,8),
            nn.BatchNorm1d(8,momentum=0.01, eps=1e-03),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(8,128),
            nn.BatchNorm1d(128,momentum=0.01, eps=1e-03),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.BatchNorm1d(128,momentum=0.01, eps=1e-03),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.BatchNorm1d(128,momentum=0.01, eps=1e-03),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.BatchNorm1d(128,momentum=0.01, eps=1e-03),
            nn.ReLU(),
            nn.Linear(128,self.input_dim),
            nn.Sigmoid()   # Sigmoid 활성화 함수 추가
        )

    def forward(self, x):
        z = self.encoder(x.view(-1, self.input_dim))
        return self.decoder(z), z