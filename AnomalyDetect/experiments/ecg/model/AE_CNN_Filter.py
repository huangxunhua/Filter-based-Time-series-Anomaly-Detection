import time,os,sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .network import weights_init
from metric import evaluate

from structure_build import generate_layer_parameter_list
import torch.nn.functional as F
from dataloader.UCR_dataloader import EM_FK


dirname=os.path.dirname
sys.path.insert(0,dirname(dirname(os.path.abspath(__file__))))

#from .plotUtil import  save_ts_heatmap_2D, save_ts_heatmap_1D, save_ts_heatmap_1D_Batch




class SampaddingConv1D_BN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(SampaddingConv1D_BN, self).__init__()
        self.padding = nn.ConstantPad1d((int((kernel_size - 1) / 2), int(kernel_size / 2)), 0)
        self.conv1d = torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
        self.bn = nn.BatchNorm1d(num_features=out_channels)

    def forward(self, X):
        X = self.padding(X)
        X = self.conv1d(X)
        X = self.bn(X)
        return X


class build_layer_with_layer_parameter(nn.Module):
    def __init__(self, layer_parameters):
        super(build_layer_with_layer_parameter, self).__init__()
        self.conv_list = nn.ModuleList()

        for i in layer_parameters:
            conv = SampaddingConv1D_BN(i[0], i[1], i[2])
            self.conv_list.append(conv)

    def forward(self, X):

        conv_result_list = []
        for conv in self.conv_list:
            conv_result = conv(X)
            conv_result_list.append(conv_result)

        result = F.relu(torch.cat(tuple(conv_result_list), 1))
        return result


class OS_CNN(nn.Module):
    def __init__(self, layer_parameter_list):
        super(OS_CNN, self).__init__()
        self.layer_parameter_list = layer_parameter_list
        self.layer_list = []

        for i in range(len(layer_parameter_list)):     # 23+23+2
            layer = build_layer_with_layer_parameter(layer_parameter_list[i])
            self.layer_list.append(layer)

        self.net = nn.Sequential(*self.layer_list)

        self.averagepool = nn.AdaptiveAvgPool1d(1)

        self.out_put_channel_numebr = 0
        for final_layer_parameters in layer_parameter_list[-1]:
            self.out_put_channel_numebr = self.out_put_channel_numebr + final_layer_parameters[1]

    def forward(self, input):  # X:(10,1,96)    (64,1,320)
        X = self.net(input)  # X:(10,200,96)    (64,46,320)
        X = self.averagepool(X)  # X:(10,200,1)   (64,46,1)

        return X  # X:(10,2)



class Decoder(nn.Module):
    def __init__(self, ngpu, opt):
        super(Decoder, self).__init__()
        self.ngpu = ngpu

        self.deconv1 = nn.Sequential(
            nn.ConvTranspose1d(opt.nz, opt.ngf * 4, 10, 1, 0, bias=False),  # (batch_size, opt.nz, 1) -> (batch_size, opt.ngf * 4, 10)
            nn.BatchNorm1d(opt.ngf * 4),
            nn.ReLU(True)
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose1d(opt.ngf * 4, opt.ngf * 2, 4, 2, 1, bias=False), # (batch_size, opt.ngf * 4, 10) -> (batch_size, opt.ngf * 2, 20)
            nn.BatchNorm1d(opt.ngf * 2),
            nn.ReLU(True),
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose1d(opt.ngf * 2, opt.ngf, 4, 2, 1, bias=False), # (batch_size, opt.ngf * 2, 20) -> (batch_size, opt.ngf, 40)
            nn.BatchNorm1d(opt.ngf),
            nn.ReLU(True),
        )

        self.deconv4 = nn.Sequential(
            nn.ConvTranspose1d(opt.ngf, opt.nc, 4, 2, 1, bias=False), # (batch_size, opt.ngf, 40) -> (batch_size, opt.nc, 80)
            #nn.BatchNorm1d(opt.nc),
            nn.Tanh(),
        )
        self.fc = nn.Sequential(
            nn.Linear(80, opt.isize), # (batch_size, opt.nc, 80) -> (batch_size, opt.nc, opt.isize)
            #nn.LeakyReLU(0.2, inplace=True),

        )

    def forward(self, z):
        out = self.deconv1(z)
        out = self.deconv2(out)
        out = self.deconv3(out)
        out = self.deconv4(out)
        out = self.fc(out)

        return out




class AE_CNN(nn.Module):

    def __init__(self, opt):
        super(AE_CNN, self).__init__()
        opt.isize

        #self.encoder1 = Encoder(opt.ngpu, opt, opt.nz)

        self.Max_kernel_size = 89  # 待修改
        self.paramenter_number_of_layer_list = [8 * 128, 1 * 128 * 256 + 0 * 256 * 128]
        #self.paramenter_number_of_layer_list = [8*128]
        self.receptive_field_shape = min(int(opt.isize / 4), self.Max_kernel_size)
        self.layer_parameter_list = generate_layer_parameter_list(1, self.receptive_field_shape,
                                                                  self.paramenter_number_of_layer_list)
        output_channel_number = 0
        for final_layer_parameters in self.layer_parameter_list[-1]:
            output_channel_number += final_layer_parameters[1]
        opt.nz = output_channel_number

        self.encoder1 = OS_CNN(self.layer_parameter_list)

        self.decoder = Decoder(opt.ngpu, opt)
        self.signal_length = opt.signal_length

    def forward(self, x_noisy): # （64，1，320）
        # latent_i = self.encoder1(x[:, :, :self.signal_length[0]])  #（64，50，1）
        latent_i = self.encoder1(x_noisy)  #（64，50，1）

        gen_x = self.decoder(latent_i)  #（64，1，320）
        return gen_x, latent_i


class ModelTrainer(nn.Module):

    def __init__(self, opt, dataloader, device):
        super(ModelTrainer, self).__init__()
        self.niter=opt.niter
        self.dataset=opt.dataset
        self.model = opt.model
        self.outf=opt.outf
        self.normal_idx = opt.normal_idx
        self.seed = opt.seed

        self.dataloader = dataloader
        self.device = device
        self.opt=opt

        self.batchsize = opt.batchsize
        self.nz = opt.nz
        self.niter = opt.niter

        self.pre_niter = opt.pre_niter

        self.G = AE_CNN(opt).to(device)
        self.G.apply(weights_init)

        self.bce_criterion = nn.BCELoss()
        self.mse_criterion=nn.MSELoss()
        self.l1_loss = nn.L1Loss()


        self.optimizerG = optim.Adam(self.G.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

        self.total_steps = 0
        self.cur_epoch=0

        self.input = torch.empty(size=(self.opt.batchsize, self.opt.nc, self.opt.isize), dtype=torch.float32, device=self.device)
        self.input_noisy = torch.empty(size=(self.opt.batchsize, self.opt.nc, self.opt.isize), dtype=torch.float32, device=self.device)

        self.label = torch.empty(size=(self.opt.batchsize,), dtype=torch.float32, device=self.device)
        self.gt = torch.empty(size=(opt.batchsize,), dtype=torch.long, device=self.device)
        self.fixed_input = torch.empty(size=(self.opt.batchsize, self.opt.nc, self.opt.isize), dtype=torch.float32, device=self.device)

        self.latent_i = None

        self.out_g = None
        self.err_g_adv = None
        self.err_g_rec = None
        self.err_g = None

        self.fake_all = None
        self.input_all = None
        self.latent_all = None
        self.mse_mean = None

        self.tsne_pred = None
        self.tsne_true = None
        self.tsne_latent = None


    def compute(self, a):
        size = a.shape[0]
        return (np.trace(a) / size) * np.eye(size)


    def pre_train(self):


        print("PreTrain for EMKF.")
        start_time = time.time()

        for epoch in range(self.pre_niter):

            self.cur_epoch+=1

            self.train_epoch()

            print( "EPOCH [{}]  \t Loss{:.4f}".format(self.cur_epoch, self.err_g_rec))

        # specify parameters
        initial_state_covariance = 1
        observation_covariance = 1
        initial_value_guess = 0
        transition_matrix = 1
        transition_covariance = 0.01

        KF = EM_FK(initial_value_guess, initial_state_covariance, observation_covariance, transition_covariance, transition_matrix)

        latent_batch = torch.mean(self.latent_i, axis=0)
        latent_batch = latent_batch.view(-1).data.cpu().numpy()

        KF_EM = KF.ft.em(X=latent_batch, n_iter=10)

        TC = self.compute(KF_EM.transition_covariance)
        OC= self.compute(KF_EM.observation_covariance)
        ISC = self.compute(KF_EM.initial_state_covariance)
        TM = self.compute(KF_EM.transition_matrices)
        ISM = [np.abs(KF_EM.initial_state_mean)]


        KF_EM = EM_FK(initial_value_guess, initial_state_covariance, observation_covariance, TC, transition_matrix)   #observation_covariance 影响最大


        return KF_EM#



    def train_epoch(self):

        epoch_start_time = time.time()
        self.G.train()
        epoch_iter = 0
        for i,data in enumerate(self.dataloader["train"], 0):
            self.total_steps += self.opt.batchsize
            epoch_iter += 1
            self.set_input(data)
            self.optimize()


    def set_input(self, input):
        with torch.no_grad():
            self.input.resize_(input[0].size()).copy_(input[0])
            self.input_noisy.resize_(input[1].size()).copy_(input[1])
            self.gt.resize_(input[2].size()).copy_(input[2])

            # fixed input for view
            if self.total_steps == self.opt.batchsize:
                self.fixed_input.resize_(input[0].size()).copy_(input[0])



    def test(self):
        '''
        test by auc value
        :return: auc
        '''
        y_true, y_pred = self.predict(self.dataloader["test"])
        rocprc, rocauc, best_th, best_f1 = evaluate(y_true, y_pred)
        return rocprc, rocauc, best_th, best_f1



    def predict(self,dataloader_,scale=True):

        with torch.no_grad():
            self.an_scores = torch.zeros(size=(len(dataloader_.dataset),), dtype=torch.float32, device=self.device)
            self.gt_labels = torch.zeros(size=(len(dataloader_.dataset),), dtype=torch.long,    device=self.device)
            self.latent_i  = torch.zeros(size=(len(dataloader_.dataset), self.opt.nz), dtype=torch.float32, device=self.device)
            self.dis_feat = torch.zeros(size=(len(dataloader_.dataset), self.opt.ndf*16*10), dtype=torch.float32,
                                        device=self.device)
            for i, data in enumerate(dataloader_, 0):

                self.set_input(data)
                self.fake, latent_i = self.G(self.input_noisy)



                # if self.cur_epoch % 10000 == 0:
                #     save_ts_heatmap_1D(self.input.detach().cpu().numpy(), self.fake.detach().cpu().numpy(),
                #                        self.gt.detach().cpu().numpy(), 'AE_CNN_Filter_Test_{}_{}_{}'.format(self.cur_epoch, i, self.normal_idx), 0.2)


                error = torch.mean(torch.pow((self.input.view(self.input.shape[0], -1) - self.fake.view(self.fake.shape[0], -1)), 2),dim=1)

                self.an_scores[i*self.opt.batchsize : i*self.opt.batchsize+error.size(0)] = error.reshape(error.size(0))
                self.gt_labels[i*self.opt.batchsize : i*self.opt.batchsize+error.size(0)] = self.gt.reshape(error.size(0))
                self.latent_i [i*self.opt.batchsize : i*self.opt.batchsize+error.size(0), :] = latent_i.reshape(error.size(0), self.opt.nz)

            # Scale error vector between [0, 1]
            if scale:
                self.an_scores = (self.an_scores - torch.min(self.an_scores)) / (torch.max(self.an_scores) - torch.min(self.an_scores))

            y_true = self.gt_labels.cpu().numpy()
            y_pred = self.an_scores.cpu().numpy()
            self.tsne_latent = self.latent_i.cpu().numpy()

            return y_true, y_pred


    def optimize(self):
        self.G.zero_grad()
        self.fake, self.latent_i = self.G(self.input_noisy)  # G的生成结果
        self.err_g_rec = self.mse_criterion(self.fake, self.input)  # constrain x' to look like x

        self.err_g = self.err_g_rec
        self.err_g.backward()
        self.optimizerG.step()


