import torch
import torch.nn as nn
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
    def forward(self, Y, x1,a1,x2,a2):
        s1,s2 = self.encoder(Y, x1,a1,x2,a2)
        rec_Y = self.decoder(s1,x1,a1,s2,x2,a2)
        return s1,s2,rec_Y
    def compute_latent_loss(self,prior_mu_S,prior_SqSigma):
        ori_latent_loss = self.encoder.latent_loss(prior_mu_S = prior_mu_S,prior_SqSigma = prior_SqSigma)
        return ori_latent_loss

class encoder(nn.Module):
    def __init__(self):
        super(encoder, self).__init__()
        Layers = []
        for _ in range(6):
            Layers += [nn.TransformerEncoderLayer(d_model= 2 * 8,nhead=8)]
        self.sigTransformer = nn.Sequential(*Layers)

        mlp_all_input_features = 2 * 8 * 50 + 2 * (2 * 1 * 30 + 2 * 1 * 8)
        mlp_out_feartures = 6 * 1 * 20
        Layers = []
        Layers += [
            nn.Linear(in_features=mlp_all_input_features,out_features=mlp_all_input_features * 2),
            nn.BatchNorm1d(num_features=mlp_all_input_features * 2),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=mlp_all_input_features * 2,out_features=mlp_out_feartures),
            nn.BatchNorm1d(num_features=mlp_out_feartures),
            nn.ReLU(inplace=True),
        ]
        self.boneLayers = nn.Sequential(*Layers)

        mu_S_features = 4*1*20
        Layers = []
        Layers += [
            nn.Linear(in_features=mlp_out_feartures,out_features=mu_S_features),
            nn.Tanh(),
        ]
        self.mu_S_Layers = nn.Sequential(*Layers)

        log_SqSigma_S_features = 2*1*20
        Layers = []
        Layers += [
            nn.Linear(in_features=mlp_out_feartures,out_features=log_SqSigma_S_features),
        ]
        self.log_SqSigma_S_Layers = nn.Sequential(*Layers)

    def forward(self, Y, x1,a1,x2,a2):
        Y = Y.permute(3,0,1,2) #(N,2,8,50) -> (50,N,2,8)
        Y = torch.flatten(Y,start_dim=2) #(50,N,2,8) -> (50,N,16)
        Y = self.sigTransformer(Y) #(50,N,16)
        Y = Y.permute(1,0,2) #(50,N,16) -> (N,50,16)
        Y = torch.flatten(Y,start_dim=1) #(N,50,16) -> (N,50 * 16)

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
            nn.BatchNorm1d(num_features=mlp_all_input_features * 2),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=mlp_all_input_features * 2,out_features=mlp_out_features * 2),
            nn.BatchNorm1d(num_features=mlp_out_features * 2),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=mlp_out_features * 2,out_features=mlp_out_features)
        ]
        self.boneMLP = nn.Sequential(*layers)

        Layers = []
        for _ in range(6):
            Layers += [nn.TransformerEncoderLayer(d_model= 2 * 8,nhead=8)]
        self.sigTransformer = nn.Sequential(*Layers)

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
        mlp_output = mlp_output.unflatten(dim=1, sizes=[2*8, 50])  # (N,2 * 8 * 50)-> (N,2*8,50)
        mlp_output = mlp_output.permute(2,0,1) #(N,2*8,50) -> (50,N,2*8)

        tsf_output = self.sigTransformer(mlp_output) #(50,N,2*8)
        tsf_output = tsf_output.unflatten(dim=2,sizes=[2,8]) #(50,N,2*8) -> (50,N,2,8)
        Y = tsf_output.permute(1,2,3,0) #(50,N,2,8) -> (N,2,8,50)
        Y = self.finalCNN(Y)
        return Y



















