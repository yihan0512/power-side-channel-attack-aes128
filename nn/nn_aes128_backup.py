import pandas as pd
import numpy as np
import scipy.io as sio
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
from torch.autograd import Variable


batchSize = 1000
learnRate = 0.001
numSample = 9000 # total number of signals used for the profiling phase
numValidate = 1000 # size of the validation set
numTest = 500 # number of signals for the attack phase
lenSignal = 4096 # power signal length
fc = 2530 # input dim of the fully connected layer
ep = 20 # number of epoches for training
subkeyByte = 0 # index of the attacked subkey byte

sbox = [
   0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5,
   0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
   0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0,
   0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
   0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC,
   0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
   0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A,
   0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
   0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0,
   0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
   0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B,
   0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
   0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85,
   0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
   0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5,
   0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
   0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17,
   0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
   0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88,
   0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
   0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C,
   0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
   0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9,
   0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
   0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6,
   0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
   0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E,
   0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
   0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94,
   0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
   0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68,
   0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16
] # sbox for aes128

def hw(xx):
    return np.binary_repr(xx).count('1')

def sbox_power(p, k):
    return hw(sbox[(p^k) % 256])


def prepareData():
#    print('reading trace...')
#    trace = sio.loadmat('/freespace/local/yh482/trace.mat')
#    trace = trace['trace']
    print('reading trace...')
    traceframe = pd.read_csv('/freespace/local/yh482/AES128.txt', sep=',', header=None)
    trace = traceframe.values
    trace = trace[:, 1:].T
    print('reading text...')
    text = sio.loadmat('/freespace/local/yh482/text.mat')
    text = text['text']
    print('reading test trace...')
    traceframe = pd.read_csv('/freespace/local/yh482/AES128_Test.txt', sep=',', header=None)
    trace_test = traceframe.values
    trace_test = trace_test[:, 1:].T

    key = [202, 254, 186, 190, 222, 173, 190, 239, 0, 1, 2, 3, 4, 5, 6, 7]
    pwr_model = np.vectorize(sbox_power)
    
    X_train = torch.Tensor(trace[:numSample-numValidate, :lenSignal])
    X_train = torch.unsqueeze(X_train, dim=1)
    y_train = torch.LongTensor(pwr_model(text[:numSample-numValidate, subkeyByte], key[subkeyByte]))
    
    X_validate = torch.Tensor(trace[numSample-numValidate:numSample, :lenSignal])
    X_validate = torch.unsqueeze(X_validate, dim=1)
    y_validate = torch.LongTensor(pwr_model(text[numSample-numValidate:numSample, 0], key[subkeyByte]))
    validate_set = [X_validate, y_validate]
    
    X_test = torch.Tensor(trace_test[:numTest, :lenSignal])
    X_test = torch.unsqueeze(X_test, dim=1)
    
    Dataset = Data.TensorDataset(X_train, y_train)
    train_loader = Data.DataLoader(dataset = Dataset, batch_size = batchSize, shuffle = True)
    return train_loader, validate_set, X_test

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(1, 10, kernel_size=4)
        self.conv2 = nn.Conv1d(10, 10, kernel_size=4)
#        self.conv3 = nn.Conv1d(10, 10, kernel_size=5)
#        self.conv2 = nn.Conv1d(10, 1, kernel_size=2)
        self.BN1 = nn.BatchNorm1d(10)
#        self.BN2 = nn.BatchNorm1d(1)
        self.fc1 = nn.Linear(fc, 9)
#        self.fc2 = nn.Linear(256, 9)

    def forward(self, x):
        x = F.max_pool1d(F.relu(self.BN1(self.conv1(x))), 2)
#        x = self.BN1(x)
        x = F.max_pool1d(F.relu(self.BN1(self.conv2(x))), 2)
#        x = self.BN1(x)
        x = F.max_pool1d(F.relu(self.BN1(self.conv2(x))), 2)
#        x = self.BN1(x)
        x = F.max_pool1d(F.relu(self.BN1(self.conv2(x))), 2)
#        print(x.shape)
        x = x.view(-1, fc)
        x = self.fc1(x)
#        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def train(epoch):    
    for batch_idx, (data, target) in enumerate(train_loader):
#        hist = np.histogram(target, bins=9)
#        plt.plot(hist[1][:-1], hist[0])
#        print(hist[0])
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % (numSample/batchSize)  == 0:
#            X_val, y_val = validate_set[0], validate_set[1]
#            X_val, y_val = Variable(X_val.cuda()), Variable(y_val.cuda())
#            output_val = model(X_val)
#            pred = output_val.data.max(1, keepdim=True)[1]
#            correct = pred.eq(y_val.data.view_as(pred)).long().cpu().sum()
            pred = output.data.max(1, keepdim=True)[1]
            correct = pred.eq(target.data.view_as(pred)).long().cpu().sum()
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} training accuracy: {:.2f}%'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0], 100. * correct/numValidate))

train_loader, validate_set, X_test = prepareData()
model = Net()
model.cuda()
optimizer = optim.Adam(model.parameters(), lr=learnRate)

for epoch in range(ep):
    train(epoch)
#for param_group in optimizer.param_groups:
#   param_group['lr'] = learnRate / 10;
#for epoch in range(100):
#    train(epoch)
#for param_group in optimizer.param_groups:
#   param_group['lr'] = learnRate / 100;
#for epoch in range(100):
#    train(epoch)
output_test = model(Variable(X_test.cuda()))
output_test = output_test.data.cpu().numpy()
np.savetxt('pdf', output_test)

# Attack phase
pdf = output_test

# load data
print('reading text...')
textframe = pd.read_csv('/freespace/local/yh482/AES128_ciphertext_plaintext_Test.txt', sep=' ', header=None)
text_test = textframe.values
text_test = text_test[:, :-1]
plaintext_test = np.zeros([numTest, 32])
for i in range(numTest):
    for j in range(32):
        plaintext_test[i, j] = int(text_test[i, j], 16)

mle = np.zeros(256)
pwr_model = np.vectorize(sbox_power)
for i in range(256):
    logsum = np.zeros([numTest, 1])
    power_pattern = pwr_model(plaintext_test[:, subkeyByte].astype(int), i)
    for j in range(numTest):
        logsum[j] = pdf[j, power_pattern[j]]
    mle[i] = np.sum(logsum)

plt.figure()
plt.plot(mle)
plt.show()
print(np.argmax(mle))
