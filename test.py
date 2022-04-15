import itertools
import os.path

import numpy
import torch
import h5py
import torch.nn
import numpy as np
from scipy.io import savemat
from scipy.io import loadmat
from torch.utils.data import DataLoader
from myModule.vae_swinIR import *
from myModule.UWBDataset import *
from myModule.APES import *
from myModule.myLoss import *
from torch.utils.tensorboard import SummaryWriter



#%% init paremeters
dataset_folders = '../../data/DataBase/20210820_EMVAE/'
modelPath =  './resultModel/v1.3_EMVAE_noInv/'
modelFileName = "vae2.pth.tar"
ResultFileName = "data.mat"
writer = SummaryWriter(log_dir=os.path.join(modelPath,"logs"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LossFun = torch.nn.SmoothL1Loss().to(device)
RegularFun = torch.nn.L1Loss().to(device)
num_epochs = 1000
batchsize = 128
EM_iter_Nums = 1
loss_test_mat = np.zeros(shape=(num_epochs,1))


#%% read data from mat
testset_dict = h5py.File(name=os.path.join(dataset_folders,'testset.mat'),mode='r')
testset = np.array(testset_dict['test_dataset'])
testset = testset.swapaxes(0,3)
testset = testset.swapaxes(1,2)
testset = testset[:,0:2,:,:]

S_dict = h5py.File(name=os.path.join(dataset_folders,'S.mat'),mode='r')
S = np.array(S_dict['S'])
S = S.swapaxes(0,2)
S = torch.Tensor(S).to(device)
S_SqSigma = torch.Tensor([0.05]).to(device)

#%% Dataloader

dataloader_test = DataLoader(
    dataset=UWBDataset(data=testset),
    batch_size=batchsize,
    shuffle=True,
    num_workers=1
)
#%% init model
model = vae().to(device)
# checkpoint = torch.load(os.path.join(modelPath,modelFileName))
checkpoint = torch.load(os.path.join(modelPath,modelFileName),map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['state_dict'])

#test
with torch.no_grad():
    batchIndex = 0
    csi, x_index_1_label, a_1, x_index_2_label, a_2 = next(iter(dataloader_test))
    for csi,x_index_1_label,a_1,x_index_2_label,a_2 in dataloader_test:
        x_1 = torch.randn(csi.shape[0], 2, 1, 30)
        x_2 = torch.randn(csi.shape[0], 2, 1, 30)

        print("begin testing EM-Iter!")
        for EM_iter in range(EM_iter_Nums):
            s_1, s_2, Y_rec = model(csi, x_1, a_1, x_2, a_2)
            x_1,x_2,x_index_1,x_index_2,X1_mask,X2_mask = model.compute_XIndex(csi, a_1, s_1, a_2, s_2)

        APES_loss1 = CAPON_Loss(Y=csi,a=a_1,S=s_1,X=x_1,mask=X1_mask)
        APES_loss2 = CAPON_Loss(Y=csi,a=a_2,S=s_2,X=x_2,mask=X2_mask)
        latent_loss = model.compute_latent_loss(prior_mu_S=S, prior_SqSigma=S_SqSigma)
        regression_Loss = LossFun(Y_rec, csi)
        d1_loss = LossFun(x_index_1, x_index_1_label)
        d2_loss = LossFun(x_index_2, x_index_2_label)

        print(
            "allbatch: {}, batch:{}, latent_loss: {:.5f}, regression_Loss: {:.5f},d1_loss: {:.5f},APES_loss1: {:.5f}, d2_loss: {:.5f},APES_loss2: {:.5f}".format(
                (testset.shape[0])/batchsize, batchIndex, latent_loss.item(), regression_Loss, d1_loss, APES_loss1, d2_loss,
                APES_loss2
            ))

        X_1_complex = torch.complex(real=x_1[:,0,:,:],imag=x_1[:,1,:,:])
        X_2_complex = torch.complex(real=x_2[:,0,:,:],imag=x_2[:,1,:,:])
        S_1_complex = torch.complex(real=s_1[:,0,:,:],imag=s_1[:,1,:,:])
        S_2_complex = torch.complex(real=s_2[:,0,:,:],imag=s_2[:,1,:,:])
        Y_complex = torch.complex(real = csi[:,0,:,:],imag=csi[:,1,:,:])
        Y_1_complex, Y1_rec_complex = CAPON_filter(Y = csi,a = a_1,S = s_1, X = x_1)
        Y_2_complex, Y2_rec_complex = CAPON_filter(Y = csi,a = a_2,S = s_2, X = x_2)
        if batchIndex == 0:
            X_1_all = X_1_complex.numpy()
            X_2_all = X_2_complex.numpy()
            S_1_all = S_1_complex.numpy()
            S_2_all = S_2_complex.numpy()
            Y_all = Y_complex.numpy()
            Y_1_all = Y_1_complex.numpy()
            Y_2_all = Y_2_complex.numpy()
            Y1_rec_all = Y1_rec_complex.numpy()
            Y2_rec_all = Y2_rec_complex.numpy()
        else:
            X_1_all = np.concatenate((X_1_all,X_1_complex.numpy()),axis=0)
            X_2_all = np.concatenate((X_2_all,X_2_complex.numpy()),axis=0)
            S_1_all = np.concatenate((S_1_all,S_1_complex.numpy()),axis=0)
            S_2_all = np.concatenate((S_2_all, S_2_complex.numpy()), axis=0)
            Y_all = np.concatenate((Y_all,Y_complex.numpy()),axis= 0)
            Y_1_all = np.concatenate((Y_1_all,Y_1_complex.numpy()),axis=0)
            Y_2_all = np.concatenate((Y_2_all,Y_2_complex.numpy()),axis=0)
            Y1_rec_all = np.concatenate((Y1_rec_all,Y1_rec_complex.numpy()), axis=0)
            Y2_rec_all = np.concatenate((Y2_rec_all,Y2_rec_complex.numpy()), axis=0)
            break

        batchIndex += 1

    matdict = {
                'X_1': X_1_all,
                'X_2': X_2_all,
                'S_1': S_1_all,
                'S_2': S_2_all,
                'Y': Y_all,
                'Y_1':Y_1_all,
                'Y_2':Y_2_all,
                'Y1_rec':Y1_rec_all,
                'Y2_rec':Y2_rec_all,
               }
    FileName = os.path.join(modelPath,ResultFileName)
    if not os.path.exists(FileName):
        os.mknod(FileName)
    f = open(FileName,mode='w')
    savemat(FileName, matdict)


