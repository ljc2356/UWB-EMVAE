import math
import torch
import torch.nn as nn
from myModule.APES import *
import torch.nn.functional as F

def APES_Loss(Y,a,S,X):
    Y = torch.complex(real=Y[:,0,:,:],imag=Y[:,1,:,:]).to(Y.device) #(N,2,8,50) ->  complex (N,8,50)
    a = torch.complex(real=a[:,0,:,:],imag=a[:,1,:,:]).to(a.device) #(N,2,8,1) -> complex (N,8,1)
    S = torch.complex(real=S[:,0,:,:],imag=S[:,1,:,:]).to(S.device) #(N,2,1,20) -> complex (N,1,20)
    X = torch.complex(real=X[:,0,:,:],imag=X[:,1,:,:]).to(X.device) #(N,2,1,30) -> complex (N,1,30)

    X = X.permute(0,2,1) #complex(N,1,30) -> complex(N,30,1)
    Y = Y.permute(0,2,1) #complex (N,8,50) -> complex (N,50,8)
    S = S.permute(0,2,1) #complex (N,1,20) -> complex (N,20,1)
    SMat = genSMat(S) #complex (N,50,30)


    R = Y.conj().permute(0, 2, 1) @ Y  # (N,8,8) = (N,8,50) * (N,50,8)
    temp = a.conj().permute(0,2,1) @ torch.linalg.inv(R) @ a  #(N,1,1)
    #(N,1,1) = (N,1,8) * (N,8,8) * (N,8,50) * (N,50,30) * (N,30,1)
    Lambda = (1 - 2 * a.conj().permute(0,2,1) @ torch.linalg.inv(R) @ Y.conj().permute(0, 2, 1) @ SMat @ X) / temp
    #(N,8,1) = (N,8,50) * (N,50,30) * (N,30,1)   + (N,1,1) * (N,8,1)
    omega = torch.linalg.inv(R) @ (2*Y.conj().permute(0, 2, 1) @ SMat @ X + Lambda * a)

    #(N,50,1) = (N,50,8) * (N,8,1) - (N,50,30) * (N,30,1)
    res = Y @ omega - SMat @ X
    #(N,1,1) = (N,1,50) * (N,50,1)
    loss = torch.abs(res.conj().permute(0,2,1)@ res)
    meanLoss = torch.mean(loss)
    return meanLoss

def CAPON_Loss(Y,a,S,X,mask):
    Y = torch.complex(real=Y[:,0,:,:],imag=Y[:,1,:,:]).to(Y.device) #(N,2,8,50) ->  complex (N,8,50)
    a = torch.complex(real=a[:,0,:,:],imag=a[:,1,:,:]).to(a.device) #(N,2,8,1) -> complex (N,8,1)
    S = torch.complex(real=S[:,0,:,:],imag=S[:,1,:,:]).to(S.device) #(N,2,1,20) -> complex (N,1,20)
    X = torch.complex(real=X[:,0,:,:],imag=X[:,1,:,:]).to(X.device) #(N,2,1,30) -> complex (N,1,30)

    X = X.permute(0,2,1) #complex(N,1,30) -> complex(N,30,1)
    Y = Y.permute(0,2,1) #complex (N,8,50) -> complex (N,50,8)
    S = S.permute(0,2,1) #complex (N,1,20) -> complex (N,20,1)
    SMat = genSMat(S) #complex (N,50,30)

    R = Y.conj().permute(0, 2, 1) @ Y  # (N,8,8) = (N,8,50) * ( N,50,8)
    Q = R
    # (N,1,1) = (N,1,8) * (N,8,8) * (N,8,1)
    temp = a.conj().permute(0,2,1) @ torch.linalg.inv(Q) @ a
    w = torch.linalg.inv(Q) @ a  #(N,8,1)
    w = w/temp #(N,8,1)

    #(N,50,1) = (N,50,8) * (N,8,1) - (N,50,30) * (N,30,1)
    res = Y @ w * mask.unsqueeze(dim=2) - SMat @ X
    #(N,1,1) = (N,1,50) * (N,50,1)
    loss = torch.abs(res.conj().permute(0,2,1)@ res)
    meanLoss = torch.mean(loss)
    return meanLoss

def CAPON_filter(Y,a,S,X):
    Y = torch.complex(real=Y[:,0,:,:],imag=Y[:,1,:,:]).to(Y.device) #(N,2,8,50) ->  complex (N,8,50)
    a = torch.complex(real=a[:,0,:,:],imag=a[:,1,:,:]).to(a.device) #(N,2,8,1) -> complex (N,8,1)
    S = torch.complex(real=S[:,0,:,:],imag=S[:,1,:,:]).to(S.device) #(N,2,1,20) -> complex (N,1,20)
    X = torch.complex(real=X[:,0,:,:],imag=X[:,1,:,:]).to(X.device) #(N,2,1,30) -> complex (N,1,30)

    X = X.permute(0,2,1) #complex(N,1,30) -> complex(N,30,1)
    Y = Y.permute(0,2,1) #complex (N,8,50) -> complex (N,50,8)
    S = S.permute(0,2,1) #complex (N,1,20) -> complex (N,20,1)
    SMat = genSMat(S) #complex (N,50,30)

    R = Y.conj().permute(0, 2, 1) @ Y  # (N,8,8) = (N,8,50) * ( N,50,8)
    Q = R
    # (N,1,1) = (N,1,8) * (N,8,8) * (N,8,1)
    temp = a.conj().permute(0,2,1) @ torch.linalg.inv(Q) @ a
    w = torch.linalg.inv(Q) @ a  #(N,8,1)
    w = w/temp #(N,8,1)

    #(N,50,1) = (N,50,8) * (N,8,1) - (N,50,30) * (N,30,1)
    Y_filter = Y @ w
    Y_rec = SMat @ X
    return Y_filter, Y_rec


def entropyLoss(X,eps = 1e-8):
    #X (N,2,1,30)
    X = torch.complex(real=X[:,0,:,:],imag=X[:,1,:,:]).to(X.device) #(N,2,1,30) -> complex (N,1,30)
    X_abs = torch.abs(X) #(N,1,30)
    X_sum = torch.sum(X_abs,dim=2,keepdim=True) #(N,1,1)
    X_prop = X_abs / X_sum + eps
    # X_prop = F.softmax(X_abs,dim=2)  #(N,1,30)
    temp = -1 * X_prop * torch.log(X_prop) #(N,1,30)
    return torch.mean(torch.sum(temp,dim=2))



