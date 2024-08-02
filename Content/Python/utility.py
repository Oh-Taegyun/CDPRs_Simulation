import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #  torch.cuda.is_available() GPU를 사용가능하면 True, 아니라면 False를 리턴

def Inverse_Kinematics(a, b, X):
    # a: 풀리의 위치 벡터
    # b: 엔드이펙터의 케이블 연결 위치 벡
    # X : [x, y, z, alpha, beta, gamma]
    alpha, beta, gamma = X[3], X[4], X[5]

    # RA 계산
    RA = torch.stack([
        torch.stack([torch.cos(alpha) * torch.cos(beta), -torch.sin(alpha) * torch.cos(gamma) + torch.cos(alpha) * torch.sin(beta) * torch.sin(gamma), torch.sin(alpha) * torch.sin(gamma) + torch.cos(alpha) * torch.sin(beta) * torch.cos(gamma)]),
        torch.stack([torch.cos(alpha) * torch.sin(beta), torch.cos(alpha) * torch.cos(gamma) + torch.sin(alpha) * torch.sin(beta) * torch.sin(gamma), -torch.cos(alpha) * torch.sin(gamma) + torch.cos(alpha) * torch.sin(beta) * torch.sin(gamma)]),
        torch.stack([-torch.sin(beta), torch.cos(beta) * torch.sin(gamma), torch.cos(beta) * torch.cos(gamma)])
    ])


    # 나머지 계산
    end_position = X[:3]

    a = a.float()
    RA = RA.float()
    L = a.view(3,-1) - end_position.view(3,-1) - RA @ b.view(3,-1)
    lengths = torch.linalg.norm(L.T, ord=2, axis=1)

    return L, lengths




