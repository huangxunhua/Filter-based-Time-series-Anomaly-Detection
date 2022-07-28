
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
from options import Options
#from dataloader.UCR_dataloader import load_data
from dataloader.UCR_dataloader import load_data, EM_FK

import numpy as np

#nohup /home/tcr/anaconda3/bin/python3.8  /home/tcr/storage/HXH/MultiModal/experiments/ecg/main.py >/dev/null 2>&1 &
#nohup /home/g817_u2/anaconda3/envs/torch1.4/bin/python3.6  /home/g817_u2/XunHua/MultiModal/experiments/ecg/main.py >/dev/null 2>&1 &
#nohup /home/tcr/anaconda3/envs/tf1.14/bin/python3.6  /home/tcr/storage/HXH/MultiModal/experiments/ecg/main.py >/dev/null 2>&1 &


device = torch.device("cuda:0" if
torch.cuda.is_available() else "cpu")
#torch.cuda.set_per_process_memory_fraction(0.05, 1)

opt = Options().parse()
#os.environ['CUDA_LAUNCH_BLOCKING'] = 1


if opt.model == 'AE_CNN_Filter':
    from model.AE_CNN_Filter import ModelTrainer

else:
    raise Exception("no this model_eeg :{}".format(opt.model))



DATASETS_NAME={

     'ProximalPhalanxOutlineCorrect': 2,

}
SEEDS=[
    1,2,3,4,5
]

if __name__ == '__main__':

    results_dir='./log1'
    #results_dir = '/home/tcr/storage/HXH/MultiModal/experiments/ecg/log10'
    #results_dir = '/home/g817_u2/XunHua/MultiModal/experiments/ecg/log10'

    opt.outf = results_dir
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    if opt.model in ['AE_CNN_noisy_multi', 'AE_CNN_Noisy']:

        file2print = open('{}/results_{}_{}-{}.log'.format(results_dir, opt.model, opt.NoisyType, opt.Snr), 'a+')
        file2print_detail = open('{}/results_{}_{}-{}_detail.log'.format(results_dir, opt.model, opt.NoisyType, opt.Snr), 'a+')

    elif opt.model in ['AE_CNN_Filter']:

        file2print = open('{}/results_{}_{}.log'.format(results_dir, opt.model, opt.FilterType), 'a+')
        file2print_detail = open('{}/results_{}_{}.detail.log'.format(results_dir, opt.model, opt.FilterType), 'a+')


    else:

        file2print = open('{}/results_{}.log'.format(results_dir, opt.model), 'a+')
        file2print_detail = open('{}/results_{}_detail.log'.format(results_dir, opt.model), 'a+')

    import datetime
    print(datetime.datetime.now())
    print(datetime.datetime.now(), file=file2print)
    print(datetime.datetime.now(), file=file2print_detail)

    if opt.model in ['AE_CNN_noisy_multi', 'AE_CNN_Noisy']:

        print('NoisyType is {}\t and SNR is{}\t'.format(opt.NoisyType, opt.Snr))
        print('NoisyType is {}\t and SNR is{}\t'.format(opt.NoisyType, opt.Snr),file= file2print_detail)
        print('NoisyType is {}\t and SNR is{}\t'.format(opt.NoisyType, opt.Snr),file= file2print)

    print("Model\tDataset\tNormal_Label\tAUC_mean\tAUC_std\tAP_mean\tAP_std\tMax_Epoch", file=file2print_detail)

    print("Model\tDataset\tTest\tAUC_mean\tAUC_std\tAP_mean\tAP_std\tMax_Epoch")
    print("Model\tDataset\tTest\tAUC_mean\tAUC_std\tAP_mean\tAP_std\tMax_Epoch", file=file2print)
    file2print.flush()
    file2print_detail.flush()

    for dataset_name in list(DATASETS_NAME.keys()):

        AUCs={}
        APs={}
        MAX_EPOCHs = {}

        for normal_idx in range(DATASETS_NAME[dataset_name]):



            print("[INFO] Dataset={}, Normal Label={}".format(dataset_name, normal_idx))

            MAX_EPOCHs_seed = {}
            AUCs_seed = {}
            APs_seed = {}
            for seed in SEEDS:

                #seed = 2
                #np.random.seed(seed)
                opt.seed = seed

                opt.normal_idx = normal_idx
                dataloader, opt.isize, opt.signal_length = load_data(opt,dataset_name, None, True)
                opt.dataset = dataset_name



                #print("[INFO] Class Distribution: {}".format(class_stat))



                opt.name = "%s/%s" % (opt.model, opt.dataset)
                expr_dir = os.path.join(opt.outf, opt.name, 'train')
                test_dir = os.path.join(opt.outf, opt.name, 'test')

                if not os.path.isdir(expr_dir):
                    os.makedirs(expr_dir)
                if not os.path.isdir(test_dir):
                    os.makedirs(test_dir)
                #
                # args = vars(opt)
                # file_name = os.path.join(expr_dir, 'opt.txt')
                # with open(file_name, 'wt') as opt_file:
                #     opt_file.write('------------ Options -------------\n')
                #     for k, v in sorted(args.items()):
                #         opt_file.write('%s: %s\n' % (str(k), str(v)))
                #     opt_file.write('-------------- End ----------------\n')
                #
                # print(opt)

                print("################", dataset_name, "##################")
                print("################  Train  ##################")

                if opt.model == 'USAD':

                    model = UsadModel(opt).to(device)

                    ap_test, auc_test, epoch_max_point = training(opt.niter, model, dataloader['train'], dataloader['val'], dataloader['test'])

                elif opt.model == 'GOAD':


                    x_train, x_val, y_val, x_test, y_test, ratio_val, ratio_test = load_trans_data_ucr_affine(opt,dataset_name)
                    model = Goad(opt)  # tc_obj = tc.TransClassifierTabular(args)
                    ap_test, auc_test, epoch_max_point = model.train(x_train, x_val, y_val, x_test, y_test, ratio_val, ratio_test)

                elif opt.model == 'AE_CNN_Filter':

                    if opt.FilterType =='Kalman':

                        model = ModelTrainer(opt, dataloader, device)

                        ap_test, auc_test, epoch_max_point = model.train()

                    elif opt.FilterType =='EM-KF':

                        model = ModelTrainer(opt, dataloader, device)
                        Filter_EM = model.pre_train()

                        dataloader, opt.isize, opt.signal_length = load_data(opt, dataset_name, Filter_EM, False)

                        model = ModelTrainer(opt, dataloader, device)

                        if opt.istest == True:

                            print('Testing')

                            model.G.load_state_dict(torch.load('/home/tcr/storage/HXH/MultiModal/experiments/ecg/Model_CheckPoint/{}_{}_{}_{}.pkl'.format(opt.model, dataset_name, normal_idx, seed)))
                            ap_test, auc_test, _, _ = model.test()
                            epoch_max_point = 0

                        else:

                            ap_test, auc_test, epoch_max_point = model.train()



                else:

                    model = ModelTrainer(opt, dataloader, device)


                    if opt.istest == True:

                        print('Testing')

                        model.G.load_state_dict(torch.load('./Model_CheckPoint/{}_{}_{}_{}.pkl'.format(opt.model, dataset_name, normal_idx, seed)))

                        #print(next(model.G.parameters()))
                        ap_test, auc_test, _, _= model.test()

                        #print(auc_test)
                        epoch_max_point = 0

                    else:


                        ap_test, auc_test, epoch_max_point = model.train()


                AUCs_seed[seed] = auc_test
                APs_seed[seed] = ap_test
                MAX_EPOCHs_seed[seed] = epoch_max_point

                # End For

            MAX_EPOCHs_seed_max = round(np.max(list(MAX_EPOCHs_seed.values())), 4)
            AUCs_seed_mean = round(np.mean(list(AUCs_seed.values())), 4)
            AUCs_seed_std = round(np.std(list(AUCs_seed.values())), 4)
            APs_seed_mean = round(np.mean(list(APs_seed.values())), 4)
            APs_seed_std = round(np.std(list(APs_seed.values())), 4)

            print("Dataset: {} \t Normal Label: {} \t AUCs={}+{} \t APs={}+{} \t MAX_EPOCHs={}".format(
                dataset_name, normal_idx, AUCs_seed_mean, AUCs_seed_std, APs_seed_mean, APs_seed_std, MAX_EPOCHs_seed))

            print("{}\t{}\t{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{}".format(
                opt.model, dataset_name, normal_idx, AUCs_seed_mean, AUCs_seed_std, APs_seed_mean, APs_seed_std,
                MAX_EPOCHs_seed_max
            ), file=file2print_detail)
            file2print_detail.flush()

            AUCs[normal_idx] = AUCs_seed_mean
            APs[normal_idx] = APs_seed_mean
            MAX_EPOCHs[normal_idx] = MAX_EPOCHs_seed_max

        print("{}\t{}\tTest\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{}".format(
            opt.model, dataset_name, np.mean(list(AUCs.values())), np.std(list(AUCs.values())),
            np.mean(list(APs.values())), np.std(list(APs.values())), np.max(list(MAX_EPOCHs.values()))
        ), file=file2print)

        # print("################## {} ###################".format(dataset_name), file=file2print)
        # print("AUCs={} \n APs={}".format(AUCs, APs), file=file2print)
        # print("AUC:\n mean={}, std={}".format(round(np.mean(list(AUCs.values())), 4),
        #                                       round(np.std(list(AUCs.values())), 4)), file=file2print)
        # print("AP:\n mean={}, std={}".format(round(np.mean(list(APs.values())), 4),
        #                                      round(np.std(list(APs.values())), 4)), file=file2print)
        #
        # print("@" * 30, file=file2print)

        file2print.flush()






