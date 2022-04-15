import torch
import torch.nn.functional as F
import numpy
import math

def genSMat(S):
    SMat = torch.Tensor()
    SMat = SMat.to(S.device)
    for index in range(30):
        zero_top = torch.zeros(size=(S.shape[0],index,1),dtype=S.dtype,device=S.device,requires_grad=True) #(N,index,1)
        zero_bottom = torch.zeros(size=(S.shape[0],30 - index,1),dtype=S.dtype,device=S.device,requires_grad=True) #(N,30-index,1)
        S_expand = torch.cat(tensors=(zero_top,S,zero_bottom),dim=1) #(N,50,1)
        SMat = torch.cat((SMat,S_expand),dim=2) #(N,50,30)
    return SMat



def APES(Y,a,S,soft_beta = 100):
    #(Notice) here can hold the grad_fn survived
    Y = torch.complex(real=Y[:,0,:,:],imag=Y[:,1,:,:]).to(Y.device) #(N,2,8,50) ->  complex (N,8,50)
    a = torch.complex(real=a[:,0,:,:],imag=a[:,1,:,:]).to(a.device) #(N,2,8,1) -> complex (N,8,1)
    S = torch.complex(real=S[:,0,:,:],imag=S[:,1,:,:]).to(S.device) #(N,2,1,20) -> complex (N,1,20)

    Y = Y.permute(0,2,1) #complex (N,8,50) -> complex (N,50,8)
    S = S.permute(0,2,1) #complex (N,1,20) -> complex (N,20,1)
    SMat = genSMat(S) #complex (N,50,30)
    #(N,50,50) = (N,50,30) * (N,30,50) * (N,50,30) * (N,30,50)
    Pi_SMat = SMat @ torch.linalg.inv(SMat.conj().permute(0,2,1) @ SMat) @ SMat.conj().permute(0,2,1)
    R = Y.conj().permute(0,2,1) @ Y #(N,8,8) = (N,8,50) * ( N,50,8)
    Q = R - Y.conj().permute(0,2,1) @ Pi_SMat @ Y #(N,8,8)

    #(N,1,1) = (N,1,8) * (N,8,8) * (N,8,1)
    temp = a.conj().permute(0,2,1) @ torch.linalg.inv(Q) @ a
    w = torch.linalg.inv(Q) @ a  #(N,8,1)
    w = w/temp #(N,8,1)

    #(N,30,1) = (N,30,50) @ (N,50,30) @ (N,30,50) @ (N,50,8) @ (N,8,1)
    x = torch.linalg.inv(SMat.conj().permute(0,2,1) @ SMat) @ SMat.conj().permute(0,2,1) @ Y @ w

    x_abs = torch.abs(x) #(N,30,1)
    x_abs = x_abs * soft_beta
    x_weight = F.softmax(x_abs,dim=1) #(N,30,1)
    x_weight = x_weight.permute(0,2,1) #(N,30,1) -> (N,1,30)
    index_list = torch.Tensor(range(30)).to(x_weight.device) #(30)
    index_list = index_list.unsqueeze(0)
    index_list = index_list.unsqueeze(2) #(1,30,1)

    x_index = x_weight @ index_list #(N,1,1) = (N,1,30) * (1,30,1)
    x_index = x_index.squeeze(dim = 2) #(N,1)

    x = x.permute(0,2,1).unsqueeze(dim=1) #(N,30,1) -> (N,1,1,30)
    x_float = torch.cat((x.real,x.imag),dim=1) #(N,2,1,30)
    return x.detach(),x_float.detach(),x_index

def CAPON(Y,a,S,soft_beta = 100):
    #(Notice) here can hold the grad_fn survived
    Y = torch.complex(real=Y[:,0,:,:],imag=Y[:,1,:,:]).to(Y.device) #(N,2,8,50) ->  complex (N,8,50)
    a = torch.complex(real=a[:,0,:,:],imag=a[:,1,:,:]).to(a.device) #(N,2,8,1) -> complex (N,8,1)
    S = torch.complex(real=S[:,0,:,:],imag=S[:,1,:,:]).to(S.device) #(N,2,1,20) -> complex (N,1,20)

    Y = Y.permute(0,2,1) #complex (N,8,50) -> complex (N,50,8)
    S = S.permute(0,2,1) #complex (N,1,20) -> complex (N,20,1)
    SMat = genSMat(S) #complex (N,50,30)

    #(N,50,50) = (N,50,30) * (N,30,50) * (N,50,30) * (N,30,50)
    Pi_SMat = SMat @ torch.linalg.inv(SMat.conj().permute(0,2,1) @ SMat) @ SMat.conj().permute(0,2,1)
    R = Y.conj().permute(0, 2, 1) @ Y  # (N,8,8) = (N,8,50) * ( N,50,8)
    Q = R

    # (N,1,1) = (N,1,8) * (N,8,8) * (N,8,1)
    temp = a.conj().permute(0,2,1) @ torch.linalg.inv(Q) @ a
    w = torch.linalg.inv(Q) @ a  #(N,8,1)
    w = w/temp #(N,8,1)

    Y_focus = Y @ w #(N,50,1) = (N,50,8) * (N,8,1)
    #(N,30,1) = (N,30,50) * (N,50,30) * (N,30,50) * (N,50,1)
    x = torch.linalg.inv(SMat.conj().permute(0,2,1) @ SMat) @ SMat.conj().permute(0,2,1) @ Y_focus

    x_abs = torch.abs(x) #(N,30,1)
    x_abs = x_abs * soft_beta
    x_weight = F.softmax(x_abs,dim=1) #(N,30,1)
    x_weight = x_weight.permute(0,2,1) #(N,30,1) -> (N,1,30)
    index_list = torch.Tensor(range(30)).to(x_weight.device) #(30)
    index_list = index_list.unsqueeze(0)
    index_list = index_list.unsqueeze(2) #(1,30,1)

    x_index = x_weight @ index_list #(N,1,1) = (N,1,30) * (1,30,1)
    x_index = x_index.squeeze(dim = 2) #(N,1)

    x = x.permute(0,2,1).unsqueeze(dim=1) #(N,30,1) -> (N,1,1,30)
    x_float = torch.cat((x.real,x.imag),dim=1) #(N,2,1,30)
    return x.detach(),x_float.detach(),x_index

def LS(Y_focus,S,soft_beta = 100):
    Y_focus = torch.complex(real=Y_focus[:, 0, :, :], imag=Y_focus[:, 1, :, :]).to(Y_focus.device)  # (N,2,1,50) ->  complex (N,1,50)
    S = torch.complex(real=S[:, 0, :, :], imag=S[:, 1, :, :]).to(S.device)  # (N,2,1,20) -> complex (N,1,20)

    Y_focus = Y_focus.permute(0,2,1) #complex (N,1,50) -> complex (N,50,1)
    S = S.permute(0,2,1) #complex (N,1,20) -> complex (N,20,1)
    SMat = genSMat(S) #complex (N,50,30)

    #(N,30,1) = (N,30,50) * (N,50,30) * (N,30,50) * (N,50,1)
    x = torch.linalg.inv(SMat.conj().permute(0,2,1) @ SMat) @ SMat.conj().permute(0,2,1) @ Y_focus

    x_abs = torch.abs(x) #(N,30,1)
    x_abs = x_abs * soft_beta
    x_weight = F.softmax(x_abs,dim=1) #(N,30,1)
    x_weight = x_weight.permute(0,2,1) #(N,30,1) -> (N,1,30)
    index_list = torch.Tensor(range(30)).to(x_weight.device) #(30)
    index_list = index_list.unsqueeze(0)
    index_list = index_list.unsqueeze(2) #(1,30,1)

    x_index = x_weight @ index_list #(N,1,1) = (N,1,30) * (1,30,1)
    x_index = x_index.squeeze(dim = 2) #(N,1)

    x = x.permute(0,2,1).unsqueeze(dim=1) #(N,30,1) -> (N,1,1,30)
    x_float = torch.cat((x.real,x.imag),dim=1) #(N,2,1,30)
    return x.detach(),x_float.detach(),x_index








