import itertools
import os.path
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

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


#%% init paremeters
dataset_folders = '../../data/DataBase/20210820_EMVAE/'
modelPath =  './resultModel/v1.3_EMVAE_noInv/'
modelFileName = "vae2.pth.tar"
writer = SummaryWriter(log_dir=os.path.join(modelPath,"logs"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LossFun = torch.nn.SmoothL1Loss().to(device)
RegularFun = torch.nn.L1Loss().to(device)
RegularLambda = 1
num_epochs = 1000
EM_iter_Nums = 1
batchSize = 128
loss_test_mat = np.zeros(shape=(num_epochs,1))


#%% read data from mat
trainset_dict = h5py.File(name=os.path.join(dataset_folders,'trainset.mat'),mode='r')
trainset = np.array(trainset_dict['train_dataset'])
trainset = trainset.swapaxes(0,3)
trainset = trainset.swapaxes(1,2)
trainset = trainset[:,0:2,:,:]

testset_dict = h5py.File(name=os.path.join(dataset_folders,'testset.mat'),mode='r')
testset = np.array(testset_dict['test_dataset'])
testset = testset.swapaxes(0,3)
testset = testset.swapaxes(1,2)
testset = testset[:,0:2,:,:]

S_dict = h5py.File(name=os.path.join(dataset_folders,'S.mat'),mode='r')
S = np.array(S_dict['S'])
S = S.swapaxes(0,2)
S = torch.Tensor(S).to(device)
S_SqSigma = torch.Tensor([0.2]).to(device)

#%% Dataloader
dataloader_train = DataLoader(
    dataset=UWBDataset(data=trainset),
    batch_size = batchSize,
    shuffle = True,
    num_workers= 1
)

dataloader_test = DataLoader(
    dataset=UWBDataset(data=testset),
    batch_size= batchSize,
    shuffle=True,
    num_workers=1
)
#%% init model
model = vae().to(device)
# checkpoint = torch.load(os.path.join(modelPath,modelFileName))
# checkpoint = torch.load(os.path.join(modelPath,modelFileName),map_location=torch.device('cpu'))
# model.load_state_dict(checkpoint['state_dict'])

optimizer = torch.optim.Adam(
    params = model.parameters(),
    lr=0.001
)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.5)

#%% train
for epoch in range(num_epochs):
    batchIndex = 0
    lossSumBatchs = 0
    # csi, x_index_1_label, a_1, x_index_2_label, a_2 = next(iter(dataloader_train))
    # csi, x_index_1_label, a_1, x_index_2_label, a_2 = next(iter(dataloader_test))
    for csi,x_index_1_label,a_1,x_index_2_label,a_2 in dataloader_train:
        # print(csi.shape)
        # print(x_index_1.shape)
        # print(a_1.shape)
        x_1 = torch.randn(csi.shape[0],2,1,30)
        x_2 = torch.randn(csi.shape[0],2,1,30)
        if torch.cuda.is_available():
            csi = csi.cuda()
            x_1 = x_1.cuda()
            x_index_1_label = x_index_1_label.cuda()
            a_1 = a_1.cuda()
            x_2 = x_2.cuda()
            x_index_2_label = x_index_2_label.cuda()
            a_2 = a_2.cuda()
        # print("begin training EM-Iter!")
        for EM_iter in range(EM_iter_Nums):
            optimizer.zero_grad()
            s_1,s_2,Y_rec = model(csi,x_1,a_1,x_2,a_2)
            x_1,x_2,x_index_1,x_index_2,X1_mask,X2_mask = model.compute_XIndex(csi,a_1,s_1,a_2,s_2)
            APES_loss1 = CAPON_Loss(Y=csi, a=a_1, S=s_1, X=x_1,mask=X1_mask)
            APES_loss2 = CAPON_Loss(Y=csi, a=a_2, S=s_2, X=x_2,mask=X2_mask)
            X1_regress = RegularLambda * entropyLoss(x_1)
            X2_regress = RegularLambda * entropyLoss(x_2)

            latent_loss = model.compute_latent_loss(prior_mu_S=S,prior_SqSigma=S_SqSigma)
            regression_Loss = LossFun(Y_rec,csi)
            d1_loss = LossFun(x_index_1,x_index_1_label)
            d2_loss = LossFun(x_index_2,x_index_2_label)

            loss = regression_Loss + d1_loss + d2_loss + APES_loss1 + APES_loss2
            # loss =  regression_Loss + d1_loss + d2_loss + APES_loss1 + APES_loss2
            lossSumBatchs = lossSumBatchs + loss.item()
            loss.backward()
            optimizer.step()

            x_1 = x_1.detach()
            x_2 = x_2.detach()

        print("Train epo: {},allBatch: {:.1f}, nowB: {}, Iter:{}, Lat: {:.2f}, reg_Loss: {:.2f},d1_loss: {:.1f},APES_loss1: {:.5f},X1_reg: {:.2f}, d2_loss: {:.2f},APES_loss2: {:.5f},X2_reg: {:.2f}".format(
            epoch, trainset.shape[0]/batchSize, batchIndex,EM_iter, latent_loss.item(), regression_Loss,d1_loss,APES_loss1,X1_regress,d2_loss,APES_loss2,X2_regress
        ))
        #gradient descent
        batchIndex += 1
        # print("end training EM-Iter!")
    # %%
    lossSumBatchs = lossSumBatchs / (batchIndex + 1)*(EM_iter_Nums)  # avg training loss
    writer.add_scalar(tag='lossTrain', scalar_value=lossSumBatchs, global_step=epoch)

    # #decent learning rate every 10 epochs
    # if (epoch%3) == 0:
    #     scheduler.step()

    #test
    with torch.no_grad():
        batchIndex = 0
        lossSumBatchs = 0
        for csi,x_index_1_label,a_1,x_index_2_label,a_2 in dataloader_test:
            x_1 = torch.randn(csi.shape[0], 2, 1, 30)
            x_2 = torch.randn(csi.shape[0], 2, 1, 30)
            if torch.cuda.is_available():
                csi = csi.cuda()
                x_1 = x_1.cuda()
                x_index_1_label = x_index_1_label.cuda()
                a_1 = a_1.cuda()
                x_2 = x_2.cuda()
                x_index_2_label = x_index_2_label.cuda()
                a_2 = a_2.cuda()
            # print("begin testing EM-Iter!")
            for EM_iter in range(EM_iter_Nums):
                s_1, s_2, Y_rec = model(csi, x_1, a_1, x_2, a_2)
                x_1,x_2,x_index_1,x_index_2,X1_mask,X2_mask = model.compute_XIndex(csi, a_1, s_1, a_2, s_2)
                APES_loss1 = CAPON_Loss(Y=csi,a=a_1,S=s_1,X=x_1,mask=X1_mask)
                APES_loss2 = CAPON_Loss(Y=csi,a=a_2,S=s_2,X=x_2,mask=X2_mask)
                X1_regress = RegularLambda * entropyLoss(x_1)
                X2_regress = RegularLambda * entropyLoss(x_2)

            latent_loss = model.compute_latent_loss(prior_mu_S=S, prior_SqSigma=S_SqSigma)
            regression_Loss = LossFun(Y_rec, csi)
            d1_loss = LossFun(x_index_1, x_index_1_label)
            d2_loss = LossFun(x_index_2, x_index_2_label)
            # loss = latent_loss + regression_Loss + d1_loss + d2_loss + APES_loss1 + APES_loss2
            loss = regression_Loss + d1_loss + d2_loss + APES_loss1 + APES_loss2
            lossSumBatchs = lossSumBatchs + loss.item()
            print(
                "Test epo: {},allBatch: {:.1f}, nowB: {}, Iter:{}, Lat: {:.2f}, reg_Loss: {:.2f},d1_loss: {:.1f},APES_loss1: {:.5f},X1_reg: {:.2f}, d2_loss: {:.2f},APES_loss2: {:.5f},X2_reg: {:.2f}".format(
                    epoch, trainset.shape[0] / batchSize, batchIndex, EM_iter, latent_loss.item(), regression_Loss,
                    d1_loss, APES_loss1, X1_regress, d2_loss, APES_loss2, X2_regress
                ))
            batchIndex += 1

    lossSumBatchs = lossSumBatchs / (batchIndex + 1) * (EM_iter_Nums)  # avg training loss
    writer.add_scalar(tag='lossTest', scalar_value=lossSumBatchs, global_step=epoch)

    #save model
    loss_test_mat[epoch,0] = lossSumBatchs
    # if (epoch >=  2) & (loss_test_mat[epoch,0]<loss_test_mat[epoch-1,0]):
    torch.save({'state_dict':model.state_dict()},os.path.join(modelPath,modelFileName))
    print("Model saved!")

