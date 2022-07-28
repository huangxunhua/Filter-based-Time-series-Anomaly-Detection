

import argparse
import os
import torch

class Options():
    """Options class

    Returns:
        [argparse]: argparse containing train and test options
    """

    def __init__(self):
        ##
        #
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        '''
        ElectricDevices
        EthanolLevel
        MiddlePhalanxOutlineAgeGroup
        MixedShapesRegularTrain
        Symbols
        TwoPatterns
        '''

        ##
        # Base
        #self.parser.add_argument('--dataloader', default='ACSF1', help='run dataloader')
        self.parser.add_argument('--data',default='UCR',help='ECG or UCR')
        self.parser.add_argument('--dataloader', default='ElectricDevices', help='run dataloader')
        self.parser.add_argument('--data_ECG', default='../datasets/ECG', help='path to ECG')
        #self.parser.add_argument('--data_UCR',default='../datasets/UCRArchive_2018',help='path to UCRdataset')
        self.parser.add_argument('--data_UCR',default='/home/tcr/storage/HXH/MultiModal/experiments/datasets/UCRArchive_2018',help='path to UCRdataset')





        #/home/tcr/HXH/MultiModal/experiments/ecg

        self.parser.add_argument('--data_MI',default='../datasets/MI/')
        self.parser.add_argument('--data_EEG',default='../datasets/EEG/')
        self.parser.add_argument('--batchsize', type=int, default=64, help='input batch size')
        self.parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
        self.parser.add_argument('--isize', type=int, default=166, help='input sequence size.')

        self.parser.add_argument('--device', type=str, default='gpu', help='Device: gpu | cpu')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')

        self.parser.add_argument('--nc', type=int, default=1, help='input sequence channels')
        self.parser.add_argument('--nz', type=int, default=4, help='size of the latent z vector')
        self.parser.add_argument('--ngf', type=int, default=32)
        self.parser.add_argument('--ndf', type=int, default=32)

        self.parser.add_argument('--model', type=str, default='AE_CNN_Filter',
                                 choices=['AE_CNN_Filter'], help='choose model')
        self.parser.add_argument('--NoisyType', type=str, default='Exponential', choices=['Gussian', 'Rayleign', 'Exponential', 'Uniform', 'Poisson', 'Gamma'])
        self.parser.add_argument('--FilterType', type=str, default='EM-KF', choices=['Savitzky', 'Wiener', 'Kalman','EM-KF'])

        self.parser.add_argument('--Snr', type=float, default=5, help='the snr of noisy')
        self.parser.add_argument('--normal_idx', type=int, default=0, help='the label index of normaly')
        self.parser.add_argument('--niter', type=int, default=1000  , help='number of epochs to train for')
        self.parser.add_argument('--early_stop', type=int, default=200)
        self.parser.add_argument('--pre_niter', type=int, default=10)


        self.parser.add_argument('--eta', type=float, default=0.8, help='Raw signal reconstruction error weights.')
        self.parser.add_argument('--theta', type=float, default=0.2, help='WaveLet signal reconstruction error weights.')

        self.parser.add_argument('--outf', default='./output', help='output folder')

        self.parser.add_argument('--eps', type=float, default=1e-10, help='number of GPUs to use')
        self.parser.add_argument('--pool', type=str, default='max', help='number of GPUs to use')
        self.parser.add_argument('--normalize', type=bool, default='True')
        self.parser.add_argument('--seed', type=int, default=0)

        
        # Train
        self.parser.add_argument('--print_freq', type=int, default=50, help='frequency of showing training results on console')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')
        self.parser.add_argument('--w_adv', type=float, default=1, help='parameter')
        self.parser.add_argument('--folder', type=int, default=0, help='folder index 0-4')
        self.parser.add_argument('--n_aug', type=int, default=0, help='aug data times')

        self.parser.add_argument('--signal_length',type=list,default=[48],help='the length of wavelet signal')
        ## Test
        self.parser.add_argument('--istest',action='store_true', default=True, help='train model_eeg or test model_eeg')
        self.parser.add_argument('--threshold', type=float, default=0.05, help='threshold score for anomaly')

        self.opt = None

    def parse(self):
        """ Parse Arguments.
        """

        self.opt = self.parser.parse_args()

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        # set gpu ids
        if self.opt.device == 'gpu':
            torch.cuda.set_device(self.opt.gpu_ids[0])

        args = vars(self.opt)

        # print('------------ Options -------------')
        # for k, v in sorted(args.items()):
        #     print('%s: %s' % (str(k), str(v)))
        # print('-------------- End ----------------')

        # save to the disk


        # self.opt.name = "%s/%s" % (self.opt.model_eeg, self.opt.dataloader)
        # expr_dir = os.path.join(self.opt.outf, self.opt.name, 'train')
        # test_dir = os.path.join(self.opt.outf, self.opt.name, 'test')
        #
        # if not os.path.isdir(expr_dir):
        #     os.makedirs(expr_dir)
        # if not os.path.isdir(test_dir):
        #     os.makedirs(test_dir)
        #
        # file_name = os.path.join(expr_dir, 'opt.txt')
        # with open(file_name, 'wt') as opt_file:
        #     opt_file.write('------------ Options -------------\n')
        #     for k, v in sorted(args.items()):
        #         opt_file.write('%s: %s\n' % (str(k), str(v)))
        #     opt_file.write('-------------- End ----------------\n')
        return self.opt
