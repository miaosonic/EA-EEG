
import os
from scipy.linalg import fractional_matrix_power
from sklearn.metrics import cohen_kappa_score
gpus = [0]
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpus))
import numpy as np
import math
import random
import datetime
import scipy.io
import scipy.io as sio
import scipy.signal as signal
import pandas as pd
import mne
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.autograd import Variable
from torchsummary import summary
import torch.autograd as autograd
from torchvision.models import vgg19

import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn.init as init
from torch.nn.utils import weight_norm
from PIL import Image
import torchvision.transforms as transforms

import torch
import torch.nn.functional as F
from model import EAEEG
from torch import nn
from torch import Tensor
from PIL import Image
from einops import rearrange
from einops.layers.torch import Rearrange, Reduce
# from common_spatial_pattern import csp

import matplotlib.pyplot as plt
# from torch.utils.tensorboard import SummaryWriter
from torch.backends import cudnn

cudnn.benchmark = False
cudnn.deterministic = True

# writer = SummaryWriter('./TensorBoardX/')
now = datetime.datetime.now()

class ExP():
    def __init__(self, nsub, result_path):
        super(ExP, self).__init__()
        self.batch_size = 64
        self.n_epochs = 1200
        self.lr = 0.0002
        self.b1 = 0.5
        self.b2 = 0.999
        self.nSub = nsub
        self.result_path = result_path

        self.start_epoch = 0
        self.root = '/data/'
        self.data = '2a'# '2b'
        self.patience = 180

        self.log_write = open(os.path.join(self.result_path, "b2_model_2_subject%d.txt" % self.nSub), "w")

        self.Tensor = torch.cuda.FloatTensor
        self.LongTensor = torch.cuda.LongTensor

        self.criterion_l1 = torch.nn.L1Loss().cuda()
        self.criterion_l2 = torch.nn.MSELoss().cuda()
        self.criterion_cls = torch.nn.CrossEntropyLoss().cuda()

        self.model = EAEEG(chans=22, samples=1000, num_classes=4)
        self.model = nn.DataParallel(self.model, device_ids=[i for i in range(len(gpus))])
        self.model = self.model.cuda()
        self.rf = None
        self.target_mean=0

    # Segmentation and Reconstruction (S&R) data augmentation
    def interaug(self, timg, label):
        aug_data = []  # 初始化存储增强数据的列表
        aug_label = []  # 初始化存储增强标签的列表
        if self.data == '2a':
            classes=4
        elif self.data == '2b':
            classes =2
        for cls4aug in range(classes):  # 遍历 4 个类（类标签范围为 1 到 4）
            cls_idx = np.where(label == cls4aug + 1)  # 查找所有等于当前类（cls4aug + 1）的标签索引
            tmp_data = timg[cls_idx]  # 从 timg 中提取与当前类对应的数据
            tmp_label = label[cls_idx]  # 提取当前类对应的标签
            if self.data == '2a':
                tmp_aug_data = np.zeros((int(self.batch_size / 4), 1, 22, 1000))
            elif self.data == '2b':
                tmp_aug_data = np.zeros((int(self.batch_size / 4), 1, 3, 1000))


            # 初始化暂存增强数据的数组，形状为 (batch_size 的 1/4, 1, 22, 1000)，
            # 对应 batch_size 大小的 1/4, 单通道, 22 个通道的数据，1000 个时间步长

            for ri in range(int(self.batch_size / 4)):  # 遍历每个增强数据的批次
                for rj in range(8):  # 将数据分成 8 段进行增强
                    rand_idx = np.random.randint(0, tmp_data.shape[0], 8)
                    # 随机选择 8 个样本索引（从 tmp_data 中随机采样）

                    tmp_aug_data[ri, :, :, rj * 125:(rj + 1) * 125] = tmp_data[rand_idx[rj], :, :,
                                                                      rj * 125:(rj + 1) * 125]
                    # 将选取的随机样本的第 rj 段（每段 125 个时间步）复制到 tmp_aug_data 中
                    # tmp_aug_data 中的时间步从 rj * 125 到 (rj + 1) * 125

            aug_data.append(tmp_aug_data)  # 将增强后的数据 tmp_aug_data 添加到 aug_data 列表
            aug_label.append(tmp_label[:int(self.batch_size / 4)])  # 将对应的标签添加到 aug_label 列表

        aug_data = np.concatenate(aug_data)  # 将 aug_data 列表中的所有数据拼接成一个 numpy 数组
        aug_label = np.concatenate(aug_label)  # 同样，将 aug_label 列表中的标签拼接成一个 numpy 数组

        aug_shuffle = np.random.permutation(len(aug_data))
        # 创建一个随机排列，用于打乱增强后的数据和标签

        aug_data = aug_data[aug_shuffle, :, :]  # 按照随机排列打乱增强后的数据
        aug_label = aug_label[aug_shuffle]  # 按照随机排列打乱增强后的标签

        aug_data = torch.from_numpy(aug_data).cuda()  # 将增强的数据转换为 PyTorch 张量，并转移到 GPU 上
        aug_data = aug_data.float()  # 将张量的数据类型转换为浮点数

        aug_label = torch.from_numpy(aug_label - 1).cuda()
        # 将增强的标签转换为 PyTorch 张量，并转移到 GPU 上，同时将标签值减 1（使标签从 0 开始）

        aug_label = aug_label.long()  # 将标签数据类型转换为长整型（适合分类任务）

        return aug_data, aug_label  # 返回增强后的数据和标签 新的试验数据可能由试验 A 的片段 1，试验 B 的片段 2，试验 C 的片段 3……拼接而成。

    def get_data(self, path,data, EA=False):
        '''	Loads the dataset 2a of the BCI Competition IV
        available on http://bnci-horizon-2020.eu/database/data-sets

        Keyword arguments:
        subject -- number of subject in [1, .. ,9]
        training -- if True, load training data
                    if False, load testing data

        Return:	data_return 	numpy matrix 	size = NO_valid_trial x 22 x 1750
                class_return 	numpy matrix 	size = NO_valid_trial
        '''
        if data == '2a':
            NO_channels = 22
            NO_tests = 6 * 48
            Window_Length = 4 * 250
        elif data == '2b':
            NO_channels = 3
            NO_tests = 600
            Window_Length = 4 * 250

        class_return = np.zeros(NO_tests)
        data_return = np.zeros((NO_tests, NO_channels, Window_Length))

        NO_valid_trial = 0
        a = scipy.io.loadmat(path)

        a_data = a['data']
        for ii in range(0, a_data.size):
            a_data1 = a_data[0, ii]
            a_data2 = [a_data1[0, 0]]
            a_data3 = a_data2[0]
            a_X = a_data3[0]
            a_trial = a_data3[1]
            a_y = a_data3[2]
            a_fs = a_data3[3]
            a_classes = a_data3[4]
            a_artifacts = a_data3[5]
            a_gender = a_data3[6]
            a_age = a_data3[7]
            if self.data == '2a':
                for trial in range(0, a_trial.size):
                    if (a_artifacts[trial] == 0):
                        data_return[NO_valid_trial, :, :] = np.transpose(
                            a_X[int(a_trial[trial] + 500):(int(a_trial[trial] + 1500)), :22])
                        class_return[NO_valid_trial] = int(a_y[trial])
                        NO_valid_trial += 1
            elif self.data == '2b':
                for trial in range(0, a_trial.size):
                    if (a_artifacts[trial] == 0):
                        # 修改为提取3个通道的数据
                        data_return[NO_valid_trial, :, :] = np.transpose(
                            a_X[int(a_trial[trial] + 500):(int(a_trial[trial] + 1500)),
                            :NO_channels])  # 使用 NO_channels 而不是固定的22
                        class_return[NO_valid_trial] = int(a_y[trial])
                        NO_valid_trial += 1
        if EA == True:
            print("train begin\n")
            data_return[0:NO_valid_trial, :, :], self.rf, self.target_mean = self.EA_train(data_return[0:NO_valid_trial, :, :])
        else:
            print("test begin\n")
        return data_return[0:NO_valid_trial, :, :], class_return[0:NO_valid_trial]

    def EA_train(self, x):
        # 训练阶段的白化处理（和您之前的实现一样）
        num_samples, num_channels, num_time_points = x.shape
        cov = np.zeros((num_samples, num_channels, num_channels))
        target_mean = np.mean(x)
        # 计算每个样本的协方差矩阵
        for i in range(num_samples):
            cov[i] = np.cov(x[i])

        # 计算整体样本的平均协方差矩阵
        refEA = np.mean(cov, axis=0)

        # 为提高稳定性，添加小的正则化项
        epsilon = 1e-5
        refEA += epsilon * np.eye(num_channels)

        # 计算参考协方差矩阵的逆平方根
        sqrtRefEA = fractional_matrix_power(refEA, -0.5)

        # 对数据进行白化
        XEA = np.zeros_like(x)
        for i in range(num_samples):
            XEA[i] = np.dot(sqrtRefEA, x[i])

        return XEA, sqrtRefEA , target_mean

    def EA_test_with(self, samples, cov_matrix, mean):
        """
        对测试集样本批量进行白化操作，使用训练集的均值和协方差矩阵。
        Args:
        - samples: 当前输入的EEG信号样本批次 (shape: [batch_size, channels, time_points])
        - mean: 训练集计算出的均值 (shape: [channels, ])
        - cov_matrix: 训练集计算出的协方差矩阵 (shape: [channels, channels])

        Returns:
        - whitened_samples: 白化后的信号样本批次 (shape: [batch_size, channels, time_points])
        """

        batch_size = samples.shape[0]
        whitened_samples = np.zeros_like(samples)  # 初始化一个与输入样本形状相同的空数组，用于存储白化后的样本

        for i in range(batch_size):
            sample = samples[i]  # 取出当前批次中的第i个样本

            # 中心化
            centered_sample = sample - mean  # 对每个样本减去训练集均值
            whitening_matrix = cov_matrix
#           try:
 #               # 使用奇异值分解(SVD)计算逆平方根
 #               U, S, Vt = np.linalg.svd(cov_matrix + 1e-5 * np.eye(cov_matrix.shape[0]), full_matrices=False)
 #               whitening_matrix = np.dot(U, np.dot(np.diag(1.0 / np.sqrt(S + 1e-5)), Vt))
 #           except np.linalg.LinAlgError:
 #               # 如果协方差矩阵不正定或存在其他数值问题，可以使用替代方法或返回原样本
#                whitening_matrix = np.eye(cov_matrix.shape[0])  # 使用单位矩阵作为备用
            # 白化操作
            whitened_sample = np.dot(whitening_matrix, centered_sample)

            # 将白化后的样本存入结果数组
            whitened_samples[i] = whitened_sample

        return whitened_samples

    def get_data_filter(self, path, N, aStop):
        '''	Loads the dataset 2a of the BCI Competition IV
        available on http://bnci-horizon-2020.eu/database/data-sets
        Keyword arguments:
        subject -- number of subject in [1, .. ,9]
        training -- if True, load training data
                    if False, load testing data

        Return:	data_return 	numpy matrix 	size = NO_valid_trial x 22 x 1750
                class_return 	numpy matrix 	size = NO_valid_trial
        '''
        if self.data == '2a':
            NO_channels = 22
            NO_tests = 6 * 48
            Window_Length = 4 * 250
        elif self.data == '2b':
            NO_channels = 3
            NO_tests = 600
            Window_Length = 4 * 250

        class_return = np.zeros(NO_tests)
        data_return = np.zeros((NO_tests, NO_channels, Window_Length))

        NO_valid_trial = 0
        a = scipy.io.loadmat(path)
        a_data = a['data']
        for ii in range(0, a_data.size):
            a_data1 = a_data[0, ii]
            a_data2 = [a_data1[0, 0]]
            a_data3 = a_data2[0]
            a_X = a_data3[0]
            a_trial = a_data3[1]
            a_y = a_data3[2]
            a_fs = a_data3[3]
            a_classes = a_data3[4]
            a_artifacts = a_data3[5]
            a_gender = a_data3[6]
            a_age = a_data3[7]

            for trial in range(0, a_trial.size):
                if (a_artifacts[trial] == 0):
                    data_return[NO_valid_trial, :, :] = np.transpose(
                        a_X[int(a_trial[trial]) + 500:(int(a_trial[trial]) + 1500), :NO_channels])
                    class_return[NO_valid_trial] = int(a_y[trial])
                    NO_valid_trial += 1

        data_return = data_return[0:NO_valid_trial, :, :]
        ws = np.array([4 * 2 / 250, 40 * 2 / 250])
        sos = signal.cheby2(N, aStop, ws, 'bandpass', output='sos')
        data_return = signal.sosfilt(sos, data_return, axis=2)
    #    data_return=self.EA(data_return)
        return data_return, class_return[0:NO_valid_trial]

    def get_source_data(self, order=2, rs=10):

        # train data
        if self.data == '2a':
            path = self.root + 'A0%dT.mat' % self.nSub
        elif self.data == '2b':
            path = self.root + 'B0%dT.mat' % self.nSub
        print('path\n',path)
        self.train_data, self.train_label = self.get_data(path,self.data,EA=True)
    #    self.train_data, self.train_label = self.get_data_filter(path, order, rs)
     #   self.train_data,self.rf,self.target_mean = self.EA_train(self.train_data)

        self.train_data = np.expand_dims(self.train_data, axis=1)
        self.train_label = np.transpose(self.train_label)
        self.allData = self.train_data
        self.allLabel = self.train_label

        shuffle_num = np.random.permutation(len(self.allData))
        self.allData = self.allData[shuffle_num, :, :, :]
        self.allLabel = self.allLabel[shuffle_num]

        # test data
        if self.data == '2a':
            path = self.root + 'A0%dE.mat' % self.nSub
        elif self.data == '2b':
            path = self.root + 'B0%dE.mat' % self.nSub
        self.test_data, self.test_label = self.get_data(path,self.data,EA=False)
    #    self.test_data, self.test_label = self.get_data_filter(path, order, rs)
        self.test_data = self.EA_test_with(self.test_data,self.rf,self.target_mean)
     #   self.test_data, self.test_label = self.get_data_filter(path, order, rs)

        self.test_data = np.expand_dims(self.test_data, axis=1)
        self.test_label = np.transpose(self.test_label)

        self.testData = self.test_data
        self.testLabel = self.test_label

        # data shape: (trial, conv channel, electrode channel, time samples)
        return self.allData, self.allLabel, self.testData, self.testLabel

    def train(self):
        patience = self.patience # 提前停止的耐心轮数，可根据训练情况调整
        bestAcc_epoch = 0
        torch.autograd.set_detect_anomaly(True)

        img, label, test_data, test_label = self.get_source_data()

        img = torch.from_numpy(img)
        label = torch.from_numpy(label - 1)

        dataset = torch.utils.data.TensorDataset(img, label)
        self.dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)

        test_data = torch.from_numpy(test_data)
        test_label = torch.from_numpy(test_label - 1)
        test_dataset = torch.utils.data.TensorDataset(test_data, test_label)
        self.test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=self.batch_size,
                                                           shuffle=True)

        # Optimizers
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        # self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=self.lr, div_factor=10.0, pct_start=0.04,final_div_factor=100.0, steps_per_epoch=len(self.dataloader),epochs=self.n_epochs)
        test_data = Variable(test_data.type(self.Tensor))
        test_label = Variable(test_label.type(self.LongTensor))

        bestAcc = 0
        averAcc = 0
        num = 0
        Y_true = 0
        Y_pred = 0

        # Train the cnn model
        total_step = len(self.dataloader)
        curr_lr = self.lr

        for e in range(self.n_epochs):
            self.model.train()
            for i, (img, label) in enumerate(self.dataloader):
                img = Variable(img.cuda().type(self.Tensor))
                label = Variable(label.cuda().type(self.LongTensor))

                # data augmentation
                aug_data, aug_label = self.interaug(self.allData, self.allLabel)
                img = torch.cat((img, aug_data))
                label = torch.cat((label, aug_label))
                # img=img.permute(0,1,3,2)
                img = np.squeeze(img, axis=1)
                outputs = self.model(img)
                loss = self.criterion_cls(outputs, label)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # self.scheduler.step()

            # test process
            if (e + 1) % 1 == 0:
                self.model.eval()
                test_data = np.squeeze(test_data, axis=1)
                Cls = self.model(test_data)

                loss_test = self.criterion_cls(Cls, test_label)
                y_pred = torch.max(Cls, 1)[1]
                acc = float((y_pred == test_label).cpu().numpy().astype(int).sum()) / float(test_label.size(0))
                train_pred = torch.max(outputs, 1)[1]
                train_acc = float((train_pred == label).cpu().numpy().astype(int).sum()) / float(label.size(0))

                if e %200 == 0:
                    print('Epoch:', e,
                          'lr: %.6f' % self.optimizer.param_groups[0]["lr"],
                          '  Train loss: %.6f' % loss.detach().cpu().numpy(),
                          '  Test loss: %.6f' % loss_test.detach().cpu().numpy(),
                          '  Train accuracy %.6f' % train_acc,
                          '  Test accuracy is %.6f' % acc)

                self.log_write.write(str(e) + "    " + str(acc) + "\n")
                num = num + 1
                averAcc = averAcc + acc
                if acc > bestAcc:
                    bestAcc = acc
                    bestAcc_epoch = e
                    Y_true = test_label
                    Y_pred = y_pred
                    torch.save(self.model.module.state_dict(), os.path.join(self.result_path, f'model{self.nSub}.pth'))
                elif e - bestAcc_epoch >= patience:
                    print(f"Early stopping at epoch {e}. No improvement for {patience} epochs.")
                    break
        averAcc = averAcc / num
        print('The average accuracy is:', averAcc)
        print('The best accuracy is:', bestAcc)
        self.log_write.write('The average accuracy is: ' + str(averAcc) + "\n")
        self.log_write.write('The best accuracy is: ' + str(bestAcc) + "\n")

        return bestAcc, averAcc, Y_true, Y_pred
        # writer.close()


def main():
    seed_n_lst = [481, 343, 222, 1215, 1817, 1278, 940, 1067, 806]
    result_path = 'result'
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    best = 0
    aver = 0
    best_accuracies = []
    aver_accuracies = []
    kappa_scores = []
    result_file_path = os.path.join(result_path, "sub_result.txt")

    with open(result_file_path, "w") as result_write:
        for i, seed_n in enumerate(seed_n_lst):
            starttime = datetime.datetime.now()
            print(f'seed is {seed_n}')
            random.seed(seed_n)
            np.random.seed(seed_n)
            torch.manual_seed(seed_n)
            torch.cuda.manual_seed(seed_n)
            torch.cuda.manual_seed_all(seed_n)

            print(f'Subject {i + 1}')
            exp = ExP(i + 1, result_path)
            bestAcc, averAcc, Y_true, Y_pred = exp.train()

            # Calculate Kappa
            kappa = cohen_kappa_score(Y_true.cpu().numpy(), Y_pred.cpu().numpy())
            kappa_scores.append(kappa)

            print(f'THE BEST ACCURACY IS {bestAcc}')
            print(f'THE AVERAGE ACCURACY IS {averAcc}')
            print(f'THE KAPPA SCORE IS {kappa}')

            result_write.write(f'Subject {i + 1} : Seed is: {seed_n}\n')
            result_write.write(f'Subject {i + 1} : The best accuracy is: {bestAcc}\n')
            result_write.write(f'Subject {i + 1} : The average accuracy is: {averAcc}\n')
            result_write.write(f'Subject {i + 1} : The Kappa score is: {kappa}\n')

            endtime = datetime.datetime.now()
            duration = endtime - starttime
            print(f'subject {i + 1} duration: {duration}')
            result_write.write(f'Subject {i + 1} duration: {duration}\n')

            best += bestAcc
            aver += averAcc
            best_accuracies.append(bestAcc)
            aver_accuracies.append(averAcc)

        best /= len(seed_n_lst)
        aver /= len(seed_n_lst)
        std_best = np.std(best_accuracies)
        std_aver = np.std(aver_accuracies)
        avg_kappa = np.mean(kappa_scores)

        result_write.write(f'**The average Best accuracy is: {best}\n')
        result_write.write(f'The average Aver accuracy is: {aver}\n')
        result_write.write(f'The standard deviation of Best accuracy is: {std_best}\n')
        result_write.write(f'The standard deviation of Aver accuracy is: {std_aver}\n')
        result_write.write(f'The average Kappa score is: {avg_kappa}\n')

    print(f'The average Aver accuracy is: {aver}\n')
    print(f'**The average Best accuracy is: {best}\n')
    print(f'The standard deviation of Best accuracy is: {std_best}\n')
    print(f'The standard deviation of Aver accuracy is: {std_aver}\n')
    print(f'The average Kappa score is: {avg_kappa}\n')


if __name__ == "__main__":
    main()
