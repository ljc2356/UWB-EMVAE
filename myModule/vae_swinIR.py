import math
import sysconfig

import torch
import torch.nn as nn
import torch.nn.functional as F
from myModule.network_swir import *
from torch.autograd import Variable


def sample(mu, log_SqSigma):
    temp = torch.randn(mu.shape)
    temp = Variable(temp, requires_grad=False)
    temp = temp.to(mu.device)
    return mu + temp * torch.sqrt(torch.exp(log_SqSigma))


class vae(nn.Module):
    def __init__(self):
        super(vae, self).__init__()
        self.encoder = encoder()
        self.decoder = decoder()
        self.focusNet = focus()
        self.LocNet = locLayer()
    def forward(self, Y, x1,a1,x2,a2):
        s1,s2 = self.encoder(Y, x1,a1,x2,a2)
        rec_Y = self.decoder(s1,x1,a1,s2,x2,a2)
        return s1,s2,rec_Y
    def compute_latent_loss(self,prior_mu_S,prior_SqSigma):
        ori_latent_loss = self.encoder.latent_loss(prior_mu_S = prior_mu_S,prior_SqSigma = prior_SqSigma)
        return ori_latent_loss
    def compute_Yfocus(self,Y,a1,a2):
        Y_focus1 = self.focusNet(Y,a1)
        Y_focus2 = self.focusNet(Y,a2)
        return Y_focus1,Y_focus2
    def compute_XIndex(self,Y,a1,S1,a2,S2):
        X1, X2, X1_Index, X2_Index,X1_mask,X2_mask = self.LocNet(Y,a1,S1,a2,S2)
        return X1, X2, X1_Index, X2_Index,X1_mask,X2_mask


class encoder(nn.Module):
    def __init__(self):
        super(encoder, self).__init__()
        Layers = []
        self.sigTransformer = UWBSwinIR(is_attention=True)

        mlp_all_input_features = 2 * 8 * 50 + 2 * (2 * 1 * 30 + 2 * 1 * 8)
        mlp_out_feartures = 6 * 1 * 20
        Layers = []
        Layers += [
            nn.Linear(in_features=mlp_all_input_features,out_features=mlp_all_input_features * 2),
            # nn.BatchNorm1d(num_features=mlp_all_input_features * 2),
            nn.LayerNorm(normalized_shape=mlp_all_input_features * 2),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=mlp_all_input_features * 2,out_features=mlp_out_feartures),
            # nn.BatchNorm1d(num_features=mlp_out_feartures),
            nn.LayerNorm(normalized_shape=mlp_out_feartures),
            nn.ReLU(inplace=True),
        ]
        self.boneLayers = nn.Sequential(*Layers)

        mu_S_features = 4*1*20
        Layers = []
        Layers += [
            nn.Linear(in_features=mlp_out_feartures,out_features=mu_S_features),
            # nn.Tanh(),
        ]
        self.mu_S_Layers = nn.Sequential(*Layers)

        log_SqSigma_S_features = 2*1*20
        Layers = []
        Layers += [
            nn.Linear(in_features=mlp_out_feartures,out_features=log_SqSigma_S_features),
        ]
        self.log_SqSigma_S_Layers = nn.Sequential(*Layers)

    def forward(self, Y, x1,a1,x2,a2):
        Y = self.sigTransformer(Y) #(N,2,8,50)
        Y = torch.flatten(Y,start_dim=1) #(N,2,8,50) -> (N,50 * 16)

        x1 = torch.flatten(x1,start_dim=1) #(N,2,1,30) -> (N,2*30)
        a1 = torch.flatten(a1,start_dim=1) #(N,2,8,1)  -> (N,2*8)
        x2 = torch.flatten(x2,start_dim=1) #(N,2,1,30) -> (N,2*30)
        a2 = torch.flatten(a2,start_dim=1) #(N,2,8.1) ->  (N,2*8)

        mlp_input = torch.cat(tensors=(Y,x1,a1,x2,a2),dim=1) #(N,50 * 16 + 2* (2*30 + 2*8))
        mlp_output = self.boneLayers(mlp_input) #(N,6 * 1 * 20)

        mu_S = self.mu_S_Layers(mlp_output)  #(N,4*1*20)
        mu_S = mu_S.unflatten(dim=1,sizes=[4,1,20])   #(N,4 * 1 * 20)-> (N,4,1,20)
        log_SqSigma_S = self.log_SqSigma_S_Layers(mlp_output) #(N,2*1*20)
        log_SqSigma_S = log_SqSigma_S.unflatten(dim=1,sizes=[2,1,20]) #(N,2*1*20) -> (N,2,1,20)

        self.mu_s1 = mu_S[:,0:2,:,:] #(N,2,1,20)
        self.log_SqSigma_s1 = log_SqSigma_S[:,0,:,:].unsqueeze(dim=1) #(N,1,1,20)

        self.mu_s2 = mu_S[:,2:4,:,:] #(N,2,1,20)
        self.log_SqSigma_s2 = log_SqSigma_S[:,1,:,:].unsqueeze(dim=1)  #(N,1,1,20)

        s1 = sample(mu = self.mu_s1,log_SqSigma=self.log_SqSigma_s1)      #(N,2,1,20)
        s2 = sample(mu = self.mu_s2,log_SqSigma=self.log_SqSigma_s2)

        return s1,s2

    def latent_loss(self,prior_mu_S,prior_SqSigma): #prior_mu_S : Tensor:(2,1,20) prior_SqSigma:Tensor:float
        prior_mu_S = torch.unsqueeze(input=prior_mu_S,dim=0) #(2,1,20) -> (1,2,1,20)
        prior_mu_S = prior_mu_S.repeat(self.mu_s1.shape[0],1,1,1) #(1,2,1,20) -> (N,2,1,20)

        expend_log_SqSigma_s1 = self.log_SqSigma_s1.repeat(1,2,1,1) #(N,1,1,20) -> (N,2,1,20)
        expend_log_SqSigma_s2 = self.log_SqSigma_s2.repeat(1,2,1,1) #(N,1,1,20) -> (N,2,1,20)


        loss_s1 = 1/(2 * prior_SqSigma) * torch.mean(self.mu_s1 * self.mu_s1 + prior_mu_S * prior_mu_S - 2 * self.mu_s1 * prior_mu_S +
                                                     torch.exp(expend_log_SqSigma_s1))+\
                  1/2 * torch.mean(torch.log(prior_SqSigma / torch.exp(expend_log_SqSigma_s1))) - 1/2
        loss_s2 = 1/(2 * prior_SqSigma) * torch.mean(self.mu_s2 * self.mu_s2 + prior_mu_S * prior_mu_S - 2 * self.mu_s2 * prior_mu_S +
                                                     torch.exp(expend_log_SqSigma_s2))+\
                  1/2 * torch.mean(torch.log(prior_SqSigma / torch.exp(expend_log_SqSigma_s2))) - 1/2
                  
        return loss_s1 + loss_s2

class decoder(nn.Module):
    def __init__(self):
        super(decoder, self).__init__()

        mlp_all_input_features = 2 * 2 * 1 * 20 + 2 * (2 * 1 * 30 + 2 * 1 * 8)
        mlp_out_features = 2 * 8 * 50
        layers = []
        layers += [
            nn.Linear(in_features=mlp_all_input_features,out_features=mlp_all_input_features * 2),
            # nn.BatchNorm1d(num_features=mlp_all_input_features * 2),
            nn.LayerNorm(normalized_shape=mlp_all_input_features * 2),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=mlp_all_input_features * 2,out_features=mlp_out_features * 2),
            # nn.BatchNorm1d(num_features=mlp_out_features * 2),
            nn.LayerNorm(normalized_shape=mlp_out_features * 2),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=mlp_out_features * 2,out_features=mlp_out_features)
        ]
        self.boneMLP = nn.Sequential(*layers)

        self.sigTransformer = UWBSwinIR(is_attention=True)

        Layers = []
        Layers += [
            nn.Conv2d(in_channels=2,out_channels=2,kernel_size=1),
            nn.Tanh()
        ]
        self.finalCNN = nn.Sequential(*Layers)

    def forward(self,s1,x1,a1,s2,x2,a2):
        s1 = torch.flatten(s1,start_dim=1) #(N,2,1,20) -> (N,2*20)
        x1 = torch.flatten(x1,start_dim=1) #(N,2,1,30) -> (N,2*30)
        a1 = torch.flatten(a1,start_dim=1) #(N,2,8,1)  -> (N,2*8)

        s2 = torch.flatten(s2,start_dim=1) #(N,2,1,20) -> (N,2*20)
        x2 = torch.flatten(x2,start_dim=1) #(N,2,1,30) -> (N,2*30)
        a2 = torch.flatten(a2,start_dim=1) #(N,2,8.1) ->  (N,2*8)

        mlp_input = torch.cat(tensors=(s1, x1, a1, s2, x2, a2), dim=1)  # (N,2 * 2 * 1 * 20 + 2 * (2 * 1 * 30 + 2 * 1 * 8))
        mlp_output = self.boneMLP(mlp_input) #(N,2 * 8 * 50)
        mlp_output = mlp_output.unflatten(dim=1, sizes=[2,8, 50])  # (N,2 * 8 * 50)-> (N,2,8,50)

        Y = self.sigTransformer(mlp_output) #(N,2,8,50)
        Y = self.finalCNN(Y)
        return Y

class focus(nn.Module):
    def __init__(self):
        super(focus, self).__init__()
        layers = []
        layers += [nn.ReflectionPad2d(padding=(7, 6, 0, 0)),
                   SwinIR(img_size=64, patch_size=1, in_chans=2, window_size=4),
                   nn.AdaptiveAvgPool2d(output_size=(8, 50)),
                   ]
        self.weightNet = nn.Sequential(*layers)
    def forward(self,Y,a):
        #Y  (N,2,8,50) a (N,2,8,1)
        Y_expend = torch.cat(tensors=(Y,a),dim=3) #(N,2,8,51)
        weight = self.weightNet(Y_expend) #(N,2,8,50)
        Y_weighted = Y * weight #(N,2,8,50) * (N,2,8,50)-> (N,2,8,50)
        Y_focus = torch.sum(input=Y_weighted,dim=2,keepdim=True) #(N,2,8,50) -> (N,2,1,50)
        return Y_focus

class locLayer(nn.Module):
    def __init__(self):
        super(locLayer, self).__init__()
        inputFeatures = 2 * 8 * 50 + 2 * (2*8*1 + 2 * 1 * 20)

        layers = []
        layers += [
            nn.Linear(in_features=inputFeatures,out_features=inputFeatures * 2),
            # nn.BatchNorm1d(num_features=inputFeatures * 2),
            nn.LayerNorm(normalized_shape=inputFeatures * 2),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=inputFeatures*2,out_features=inputFeatures),
            nn.ReLU(inplace=True),
        ]
        self.MLP1 = nn.Sequential(*layers)

        middleFeatures = 10 * 4 * 1 * 30
        layers = []
        layers += [
            nn.Linear(in_features=inputFeatures,out_features=middleFeatures),
        ]
        self.MLP2 = nn.Sequential(*layers)

        layers = []
        layers += [
            # nn.BatchNorm1d(num_features=4 * 1 * 30),
            nn.LayerNorm(normalized_shape=middleFeatures),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=middleFeatures, out_features=6),
            nn.ReLU(inplace=True)
        ]
        self.MLP3 = nn.Sequential(*layers)

        layers = []
        layers += [
            # nn.BatchNorm1d(num_features=4 * 1 * 30),
            nn.LayerNorm(normalized_shape=middleFeatures),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=middleFeatures, out_features=4 * 1 * 30),
        ]
        self.MLP4 = nn.Sequential(*layers)


    def forward(self,Y,a1,S1,a2,S2,soft_beta = 1):
        Y = torch.flatten(Y,start_dim=1) #(N,2,8,50) -> (N,2*8*50)
        a1 = torch.flatten(a1,start_dim=1) #(N,2,8,1) -> (N,2*8*1)
        S1 = torch.flatten(S1,start_dim=1) #(N,2,1,20) -> (N,2*1*20)
        a2 = torch.flatten(a2,start_dim=1) #(N,2,8,1) -> (N,2*8*1)
        S2 = torch.flatten(S2,start_dim=1) #(N,2,1,20) -> (N,2*1*20)

        MLPinput = torch.cat(tensors=(Y,a1,S1,a2,S2),dim=1) #(N,2*8*50 + 2*(2*8*1 + 2*1*20))
        MLP1_out = MLPinput + self.MLP1(MLPinput)

        MLP2_out = self.MLP2(MLP1_out)

        X_flatten = self.MLP4(MLP2_out)
        X = X_flatten.unflatten(dim=1,sizes=[4,1,30])
        X1 = X[:,0:2,:,:]
        X2 = X[:,2:4,:,:]

        X_Index = self.MLP3(MLP2_out)
        X1_Index = X_Index[:,0]
        X2_Index = X_Index[:,1]

        X1_Int_Index = torch.clamp(torch.round(X1_Index),min=0,max=29).to(torch.int64) #(N)
        X2_Int_Index = torch.clamp(torch.round(X2_Index),min=0,max=29).to(torch.int64) #(N)
        X1_weight = F.one_hot(X1_Int_Index,num_classes = 30) #(N,30)
        X2_weight = F.one_hot(X2_Int_Index, num_classes = 30)  #(N,30)
        X1_mask = gen_Mask(X1_weight)
        X2_mask = gen_Mask(X2_weight)

        # X1 = X_Index[:,2:4] #(N,2)
        # X1 = X1.unflatten(dim=1,sizes=(2,1,1)) #(N,2,1,1)
        # X1 = X1.repeat(1,1,1,30) #(N,2,1,30)
        # X2 = X_Index[:,4:6] #(N,2)
        # X2 = X2.unflatten(dim=1,sizes=(2,1,1)) #(N,2,1,1)
        # X2 = X2.repeat(1,1,1,30) #(N,2,1,30)
        #
        # X1_Int_Index = torch.clamp(torch.round(X1_Index),min=0,max=29).to(torch.int64) #(N)
        # X2_Int_Index = torch.clamp(torch.round(X2_Index),min=0,max=29).to(torch.int64) #(N)
        # X1_weight = F.one_hot(X1_Int_Index,num_classes = 30) #(N,30)
        # X1_weight = X1_weight.unflatten(dim=1,sizes=(1,1,30))  # (N,1,1,30)
        # X2_weight = F.one_hot(X2_Int_Index, num_classes = 30)  #(N,30)
        # X2_weight = X2_weight.unflatten(dim=1,sizes=(1,1,30))  # (N,1,1,30)

        # X1 = X1 * X1_weight
        # X2 = X2 * X2_weight
        return X1,X2,X1_Index,X2_Index,X1_mask,X2_mask

def gen_Mask(X_oneHot):
    #X_oneHot: (N,30)
    Mask = torch.zeros(size=(X_oneHot.shape[0],50),dtype=X_oneHot.dtype) #(N,50)
    Mask = Mask.to(X_oneHot)
    for index in range(20):
        zero_left = torch.zeros(size=(X_oneHot.shape[0],index),device=X_oneHot.device,dtype=X_oneHot.dtype) #(N,index)
        zero_right = torch.zeros(size=(X_oneHot.shape[0],20-index),device= X_oneHot.device,dtype=X_oneHot.dtype) #(N,20-index)
        X_oneHot_expend = torch.cat(tensors=(zero_left,X_oneHot,zero_right),dim=1) #(N,50) = (N,index + 30 + 20 -index)
        Mask += X_oneHot_expend  #(N,50)
    return Mask


def soft_argmax(X,soft_beta = 1):
    maxIndex = X.shape[-1]
    complexX = torch.complex(real=X[:,0,:,:],imag=X[:,1,:,:]).to(X.device) #(N,2,1,30) -> complex (N,1,30)
    x_abs = torch.abs(complexX)  # (N,1,30)
    x_abs = x_abs * soft_beta

    x_weight = F.gumbel_softmax(x_abs,tau=1,hard=True,dim=2)
    # x_weight = F.softmax(x_abs,dim=2) #(N,1,30)
    index_list = torch.Tensor(range(maxIndex)).to(x_weight.device) #(30)
    index_list = index_list.unsqueeze(0)
    index_list = index_list.unsqueeze(2) #(1,30,1)

    x_index = x_weight @ index_list #(N,1,1) = (N,1,30) * (1,30,1)
    x_index = x_index.squeeze(dim = 2) #(N,1)

    return x_index

















