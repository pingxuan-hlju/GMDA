import numpy as np
from numpy import*
import torch
import torch.utils.data as Data
import torch.nn as nn
from pylab import *
import torch.nn.functional as F
from sklearn.metrics import auc

BATCH_SIZE = 1
EPOCH = 1

class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 30, kernel_size=(3, 17), stride=1, padding=1),
            nn.BatchNorm2d(30),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Conv2d(30, 60, kernel_size=(3, 17), stride=1, padding=1),
            nn.BatchNorm2d(60),
            nn.ReLU(),
            nn.MaxPool2d((1, 2)),

        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(60, 30, kernel_size=(2, 18), stride=2, padding=1),
            nn.BatchNorm2d(30),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(30, 1, kernel_size=(2, 18), stride=2, padding=1),
            nn.LeakyReLU(0.2),

        )

    def forward(self, x):
        encode = self.encoder(x)
        decode = self.decoder(encode)
        return encode, decode


class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 30, kernel_size=(3, 17), stride=1, padding=1),
            nn.BatchNorm2d(30),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Conv2d(30, 60, kernel_size=(3, 17), stride=1, padding=1),
            nn.BatchNorm2d(60),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d((1, 2)),
            nn.Conv2d(60, 20, kernel_size=(3, 16), stride=1, padding=1),
            nn.BatchNorm2d(20),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d((2, 2)),

        )

        self.fc = nn.Sequential(
            nn.Linear(20 * 1 * 130, 1600),
            nn.BatchNorm1d(1600),
            nn.LeakyReLU(0.2),
            nn.Linear(1600, 600),
            nn.BatchNorm1d(600),
            nn.LeakyReLU(0.2),
            nn.Linear(600, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.node1 = nn.Linear(input_att, output_att, bias=True)
        nn.init.xavier_normal_(self.node1.weight)
        self.h_n_parameters = nn.Parameter(torch.randn(output_att, 1))
        nn.init.xavier_normal_(self.h_n_parameters)
    def forward(self, h_n_states1 ,h_n_states2):
        length = h_n_states1.size()[2]
        batch = h_n_states1.size()[0]
        second_pad = Variable(torch.zeros(batch, h_n_states1.size()[1], length - h_n_states2.size()[2])).cuda()
        second_pad = torch.cat((h_n_states2, second_pad), dim=2)
        reslut = torch.cat((h_n_states1, second_pad), dim=1)
        temp_nodes = self.node1(reslut)
        temp_nodes = torch.tanh(temp_nodes)
        nodes_score = torch.matmul(temp_nodes, self.h_n_parameters)
        nodes_score = nodes_score.view(-1, 1, 2)
        beta = F.softmax(nodes_score, dim=2)
        y = torch.matmul(beta, reslut)
        y = y.view(-1, 60, 1, 273)
        return y

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(60, 40, kernel_size=2, stride=1, padding=1),
            nn.BatchNorm2d(40),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(40, 20, kernel_size=2, stride=1, padding=1),
            nn.BatchNorm2d(20),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(kernel_size=2),
        )

        self.fc = nn.Linear(output_cnn, 2)
        self.drop = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.fc(x)
        return output

    train_data = torch.from_numpy(train_list)
    test_data = torch.from_numpy(test_list)
    train_data = torch.unsqueeze(train_data, dim=1).type(torch.FloatTensor)
    test_data = torch.unsqueeze(test_data, dim=1).type(torch.FloatTensor)
    train_lable = train_lable.flatten()
    train_lable = torch.from_numpy(train_lable).long()
    test_lable = test_lable.flatten()
    test_lable = torch.from_numpy(test_lable).long()
    train_dataset = Data.TensorDataset(train_data, train_lable)
    train_loader = Data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataset = Data.TensorDataset(test_data, test_lable)
    test_loader = Data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    G = generator()
    D = discriminator()
    cnn = CNN()
    if torch.cuda.is_available():
        G = G.cuda()
        D = D.cuda()
        cnn = cnn.cuda()
    d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0001)
    g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0001)
    cnn_optimizer = torch.optim.Adam(cnn.parameters(), lr=0.0001)
    criterion = nn.BCELoss()
    cnn_lossfunc = nn.CrossEntropyLoss()

    for epoch in range(EPOCH):
        D_loss = 0
        G_loss = 0
        for step, (data, lable) in enumerate(train_loader):
            size = data.shape[0]
            real_data = data
            real_label = torch.ones(size, 1)
            fake_label = torch.zeros(size, 1)
            if torch.cuda.is_available():
                real_data = real_data.cuda()
                lable = lable.cuda()
                real_label = real_label.cuda()
                fake_label = fake_label.cuda()

            real_out = D(real_data)
            d_loss_real = criterion(real_out, real_label)
            real_scores = real_out

            fake_data_encode, fake_data_decode = G(real_data)
            fake_out = D(fake_data_decode)
            d_loss_fake = criterion(fake_out, fake_label)
            fake_scores = fake_out

            d_loss = d_loss_real + d_loss_fake
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()
            D_loss += d_loss.item()

            fake_data_encode, fake_data_decode = G(real_data)
            output = D(fake_data_decode)
            g_loss = criterion(output, real_label)

            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()
            G_loss += g_loss.item()

            if step % 20 == 0:
                print('Epoch: ', epoch, '| d_loss: %.4f' % d_loss.data.cpu().numpy(),
                      '| g_loss: %.4f' % g_loss.data.cpu().numpy())
        print('epoch: {}, D_Loss: {:.6f}, G_Loss: {:.6f}'
              .format(epoch, D_loss / len(train_loader), G_loss / len(train_loader)))

    for epoch in range(EPOCH):
        for step, (data, lable) in enumerate(train_loader):
            if torch.cuda.is_available():
                data = data.cuda()
                lable = lable.cuda()

            fake_data_encode, fake_data_decode = G(data)
            cnn_out = cnn(fake_data_encode)
            loss = cnn_lossfunc(cnn_out, lable)
            cnn_optimizer.zero_grad()
            loss.backward()
            cnn_optimizer.step()
            if step % 20 == 0:
                pred_lable = torch.max(cnn_out, 1)[1]
                accuracy = float((pred_lable == lable).sum()) / float(lable.size(0))
                print('Epoch: ', epoch, '| train cnn_loss: %.4f' % loss.data.cpu().numpy(),
                      '| test cnn_accuracy: %.2f' % accuracy)

    predict_list = zeros((len(miRNA_sim), len(dis_sim)))
    o = zeros((0, 2))


    for step_test, (data_test, lable_test) in enumerate(test_loader):
        if torch.cuda.is_available():
            data_test = data_test.cuda()
            lable_test = lable_test.cuda()
        test_output_encode, test_output_decode = G(data_test)
        test_output = cnn(test_output_encode)
        test_output = F.softmax(test_output, dim=1)
        o = vstack((o, test_output.detach().cpu().numpy()))
        pred_y = torch.max(test_output, 1)[1].data.squeeze().cpu().numpy()
        accuracy = float((pred_y == lable_test.data.cpu().numpy()).astype(int).sum()) / float(lable_test.size(0))
        print('test accuracy: %.6f' % accuracy)
