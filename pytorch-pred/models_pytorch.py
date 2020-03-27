import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np


def move_data_to_gpu(x, cuda):

    if 'float' in str(x.dtype):
        x = torch.Tensor(x)

    elif 'int' in str(x.dtype):
        x = torch.LongTensor(x)

    else:
        raise Exception("Error!")

    if cuda:
        x = x.cuda()

    x = Variable(x)

    return x

def init_layer(layer):
    """Initialize a Linear or Convolutional layer. 
    Ref: He, Kaiming, et al. "Delving deep into rectifiers: Surpassing 
    human-level performance on imagenet classification." Proceedings of the 
    IEEE international conference on computer vision. 2015.
    """
    
    if layer.weight.ndimension() == 4:
        (n_out, n_in, height, width) = layer.weight.size()
        n = n_in * height * width
        
    elif layer.weight.ndimension() == 2:
        (n_out, n) = layer.weight.size()

    std = math.sqrt(2. / n)
    scale = std * math.sqrt(3.)
    layer.weight.data.uniform_(-scale, scale)

    if layer.bias is not None:
        layer.bias.data.fill_(0.)


def init_bn(bn):
    """Initialize a Batchnorm layer. """
    
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


class BaselineCnn(nn.Module):
    def __init__(self, classes_num):
        super(BaselineCnn, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64,
                                kernel_size=(5, 5), stride= (1, 1),
                                padding=(2, 2), bias=False)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128,
                                   kernel_size=(5, 5), stride= (1, 1),
                                   padding=(2, 2), bias=False)

        self.fc1 = nn.Linear(128, classes_num, bias=True)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)

        self.init_weights()

    def init_weights(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_layer(self.fc1)

        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, input, return_bottleneck=False):
        (_, seq_len, mel_bins) = input.shape

        x = input.view(-1, 1, seq_len, mel_bins)
        """(samples_num, feature_maps, time_steps, freq_num)"""

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, kernel_size=(2, 2))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, kernel_size=(2, 2))

        x = F.max_pool2d(x, kernel_size=x.shape[2:])
        x = x.view(x.shape[0:2])

        x = F.log_softmax(self.fc1(x), dim=-1)

        return x
###################################################################################################
class EmbeddingLayers_Nopooling(nn.Module):
    def __init__(self, cond_layer=1):
        super(EmbeddingLayers_Nopooling, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64,
                               kernel_size=(5, 5), stride=(1, 1),
                               padding=(2, 2), bias=False)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128,
                               kernel_size=(5, 5), stride=(1, 1),
                               padding=(2, 2), bias=False)

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256,
                               kernel_size=(5, 5), stride=(1, 1),
                               padding=(2, 2), bias=False)

        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512,
                               kernel_size=(5, 5), stride=(1, 1),
                               padding=(2, 2), bias=False)

	self.condlayer = cond_layer
	if cond_layer==1:
            self.cond = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(1, 1))
	elif cond_layer==2:
            self.cond2 = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=(1, 1))
	elif cond_layer==3:
            self.cond3 = nn.Conv2d(in_channels=3, out_channels=256, kernel_size=(1, 1))
	elif cond_layer==4:
            self.cond4 = nn.Conv2d(in_channels=3, out_channels=512, kernel_size=(1, 1))

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)

        self.init_weights()

    def init_weights(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_layer(self.conv3)
        init_layer(self.conv4)
	if self.condlayer==1:
            init_layer(self.cond)
	elif self.condlayer==2:
            init_layer(self.cond2)
	elif self.condlayer==3:
            init_layer(self.cond3)
	elif self.condlayer==4:
            init_layer(self.cond4)

        init_bn(self.bn1)
        init_bn(self.bn2)
        init_bn(self.bn3)
        init_bn(self.bn4)

    def forward(self, input, device, return_layers=False):
        (_, seq_len, mel_bins) = input.shape

        x = input.view(-1, 1, seq_len, mel_bins)
        """(samples_num, feature_maps, time_steps, freq_num)"""

        device = torch.unsqueeze(torch.unsqueeze(device, 2), 3)
        device = device.expand(-1, -1, seq_len, mel_bins)

	if self.condlayer==1:
            x = F.relu(self.bn1(torch.add(self.conv1(x), self.cond(device))))
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.relu(self.bn3(self.conv3(x)))
            emb = F.relu(self.bn4(self.conv4(x)))

	elif self.condlayer==2:
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(torch.add(self.conv2(x), self.cond2(device))))
            x = F.relu(self.bn3(self.conv3(x)))
            emb = F.relu(self.bn4(self.conv4(x)))

	elif self.condlayer==3:
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.relu(self.bn3(torch.add(self.conv3(x), self.cond3(device))))
            emb = F.relu(self.bn4(self.conv4(x)))

	elif self.condlayer==4:
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.relu(self.bn3(self.conv3(x)))
            emb = F.relu(self.bn4(torch.add(self.conv4(x), self.cond4(device))))

        if return_layers is False:
            return emb

class CnnNoPooling_Max(nn.Module):
    def __init__(self, classes_num, devices_num, cond_layer):
        super(CnnNoPooling_Max, self).__init__()

        self.emb = EmbeddingLayers_Nopooling(cond_layer)
        self.fc_final = nn.Linear(512, classes_num)

        self.cnn_device = BaselineCnn(devices_num)

        self.init_weights()

    def init_weights(self):
        init_layer(self.fc_final)

    def forward(self, input):
        """(samples_num, feature_maps, time_steps, freq_num)"""

        device = self.cnn_device(input) 
        device_onehot = torch.argmax(device, dim=1)
	device_onehot = torch.eye(3)[device_onehot]
	device_onehot = move_data_to_gpu(device_onehot, True)

	x = self.emb(input, device_onehot)
        x = F.max_pool2d(x, kernel_size=x.shape[2:])
        x = x.view(x.shape[0:2])

        output = F.log_softmax(self.fc_final(x), dim=-1)

        return output, device

class CnnNoPooling_Avg(nn.Module):
    def __init__(self, classes_num, devices_num, cond_layer):
        super(CnnNoPooling_Avg, self).__init__()

        self.emb = EmbeddingLayers_Nopooling(cond_layer)
        self.fc_final = nn.Linear(512, classes_num)

        self.cnn_device = BaselineCnn(devices_num)

        self.init_weights()

    def init_weights(self):
        init_layer(self.fc_final)

    def forward(self, input):
        """(samples_num, feature_maps, time_steps, freq_num)"""

        device = self.cnn_device(input) 
        device_onehot = torch.argmax(device, dim=1)
	device_onehot = torch.eye(3)[device_onehot]
	device_onehot = move_data_to_gpu(device_onehot, True)

	x = self.emb(input, device_onehot)

        x = F.avg_pool2d(x, kernel_size=x.shape[2:])
        x = x.view(x.shape[0:2])

        output = F.log_softmax(self.fc_final(x), dim=-1)

        return output, device

class CnnNoPooling_roi(nn.Module):
    def __init__(self, classes_num, devices_num, cond_layer):
        super(CnnNoPooling_roi, self).__init__()

        self.emb = EmbeddingLayers_Nopooling(cond_layer)
        self.fc_final = nn.Linear(40960, classes_num)

        self.cnn_device = BaselineCnn(devices_num)

        self.init_weights()

    def init_weights(self):
        init_layer(self.fc_final)

    def forward(self, input):
        """(samples_num, feature_maps, time_steps, freq_num)"""

        device = self.cnn_device(input) 
        device_onehot = torch.argmax(device, dim=1)
	device_onehot = torch.eye(3)[device_onehot]
	device_onehot = move_data_to_gpu(device_onehot, True)

	x = self.emb(input, device_onehot)

        x = F.max_pool2d(x, kernel_size= (16, 16), stride=(16, 16))
        x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))

        output = F.log_softmax(self.fc_final(x), dim=-1)

        return output, device

class CnnNoPooling_roi_attention(nn.Module):
    def __init__(self, classes_num, devices_num, cond_layer):
        super(CnnNoPooling_roi_attention, self).__init__()

        self.emb = EmbeddingLayers_Nopooling(cond_layer)
        self.attention = Attention2d(
            512,
            classes_num,
            att_activation='sigmoid',
            cla_activation='log_softmax')

        self.cnn_device = BaselineCnn(devices_num)

    def init_weights(self):
        pass

    def forward(self, input):
        """(samples_num, feature_maps, time_steps, freq_num)"""
        device = self.cnn_device(input) 
        device_onehot = torch.argmax(device, dim=1)
	device_onehot = torch.eye(3)[device_onehot]
	device_onehot = move_data_to_gpu(device_onehot, True)

        x = self.emb(input, device_onehot)

        x = F.max_pool2d(x, kernel_size= (16, 16), stride=(16, 16))
        output = self.attention(x)

        return output, device

class CnnNoPooling_Attention(nn.Module):
    def __init__(self, classes_num, devices_num, cond_layer):
        super(CnnNoPooling_Attention, self).__init__()

        self.emb = EmbeddingLayers_Nopooling(cond_layer)
        self.attention = Attention2d(
            512,
            classes_num,
            att_activation='sigmoid',
            cla_activation='log_softmax')

        self.cnn_device = BaselineCnn(devices_num)

    def init_weights(self):
        pass

    def forward(self, input):
        """(samples_num, feature_maps, time_steps, freq_num)"""
        device = self.cnn_device(input) 

        device_onehot = torch.argmax(device, dim=1)
	device_onehot = torch.eye(3)[device_onehot]

	device_onehot = move_data_to_gpu(device_onehot, True)

        x = self.emb(input, device_onehot)

        output = self.attention(x)

        return output, device
#####################################################################################################
class EmbeddingLayers_atrous(nn.Module):
    def __init__(self, cond_layer=4):
        super(EmbeddingLayers_atrous, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64,
                               kernel_size=(5, 5), stride=(1, 1),  dilation=1,
                               padding=(2, 2), bias=False)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128,
                               kernel_size=(5, 5), stride=(1, 1),  dilation=2,
                               padding=(4, 4), bias=False)

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256,
                               kernel_size=(5, 5), stride=(1, 1),  dilation=4,
                               padding=(8, 8), bias=False)

        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512,
                               kernel_size=(5, 5), stride=(1, 1),  dilation=8,
                               padding=(16, 16), bias=False)

	self.condlayer = cond_layer
	if cond_layer==1:
            self.cond = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(1, 1))
	elif cond_layer==2:
            self.cond2 = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=(1, 1))
	elif cond_layer==3:
            self.cond3 = nn.Conv2d(in_channels=3, out_channels=256, kernel_size=(1, 1))
	elif cond_layer==4:
            self.cond4 = nn.Conv2d(in_channels=3, out_channels=512, kernel_size=(1, 1))

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)

        self.init_weights()

    def init_weights(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_layer(self.conv3)
        init_layer(self.conv4)

	if self.condlayer==1:
            init_layer(self.cond)
	elif self.condlayer==2:
            init_layer(self.cond2)
	elif self.condlayer==3:
            init_layer(self.cond3)
	elif self.condlayer==4:
            init_layer(self.cond4)

        init_bn(self.bn1)
        init_bn(self.bn2)
        init_bn(self.bn3)
        init_bn(self.bn4)

    def forward(self, input, device, return_layers=False):
        (_, seq_len, mel_bins) = input.shape

        x = input.view(-1, 1, seq_len, mel_bins)
        """(samples_num, feature_maps, time_steps, freq_num)"""

        device = torch.unsqueeze(torch.unsqueeze(device, 2), 3)
        device = device.expand(-1, -1, seq_len, mel_bins)

	if self.condlayer==1:
            x = F.relu(self.bn1(torch.add(self.conv1(x), self.cond(device))))
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.relu(self.bn3(self.conv3(x)))
            x = F.relu(self.bn4(self.conv4(x)))

	elif self.condlayer==2:
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(torch.add(self.conv2(x), self.cond2(device))))
            x = F.relu(self.bn3(self.conv3(x)))
            x = F.relu(self.bn4(self.conv4(x)))

	elif self.condlayer==3:
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.relu(self.bn3(torch.add(self.conv3(x), self.cond3(device))))
            x = F.relu(self.bn4(self.conv4(x)))

	elif self.condlayer==4:
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.relu(self.bn3(self.conv3(x)))
            x = F.relu(self.bn4(torch.add(self.conv4(x), self.cond4(device))))

        return x

class CnnAtrous_Max(nn.Module):
    def __init__(self, classes_num, devices_num, cond_layer):
        super(CnnAtrous_Max, self).__init__()

        self.emb = EmbeddingLayers_atrous(cond_layer)
        self.fc_final = nn.Linear(512, classes_num)
        self.cnn_device = BaselineCnn(devices_num)

        self.init_weights()

    def init_weights(self):
        init_layer(self.fc_final)

    def forward(self, input):
        """(samples_num, feature_maps, time_steps, freq_num)"""

        device = self.cnn_device(input) 

        device_onehot = torch.argmax(device, dim=1)
	device_onehot = torch.eye(3)[device_onehot]

	device_onehot = move_data_to_gpu(device_onehot, True)

	x = self.emb(input, device_onehot)

        x = F.max_pool2d(x, kernel_size=x.shape[2:])
        x = x.view(x.shape[0:2])

        x = F.log_softmax(self.fc_final(x), dim=-1)

        return x, device

class CnnAtrous_Avg(nn.Module):
    def __init__(self, classes_num, devices_num, cond_layer):
        super(CnnAtrous_Avg, self).__init__()

        self.emb = EmbeddingLayers_atrous(cond_layer)
        self.fc_final = nn.Linear(512, classes_num)
        self.cnn_device = BaselineCnn(devices_num)

        self.init_weights()

    def init_weights(self):
        init_layer(self.fc_final)

    def forward(self, input):
        """(samples_num, feature_maps, time_steps, freq_num)"""

        device = self.cnn_device(input) 

        device_onehot = torch.argmax(device, dim=1)
	device_onehot = torch.eye(3)[device_onehot]

	device_onehot = move_data_to_gpu(device_onehot, True)

	x = self.emb(input, device_onehot)

        x = F.avg_pool2d(x, kernel_size=x.shape[2:])
        x = x.view(x.shape[0:2])

        output = F.log_softmax(self.fc_final(x), dim=-1)

        return output, device

class CnnAtrous_roi(nn.Module):
    def __init__(self, classes_num, devices_num, cond_layer):
        super(CnnAtrous_roi, self).__init__()

        self.emb = EmbeddingLayers_atrous(cond_layer)
        self.fc_final = nn.Linear(40960, classes_num)
        self.cnn_device = BaselineCnn(devices_num)

        self.init_weights()

    def init_weights(self):
        init_layer(self.fc_final)

    def forward(self, input):
        """(samples_num, feature_maps, time_steps, freq_num)"""
        device = self.cnn_device(input) 

        device_onehot = torch.argmax(device, dim=1)
	device_onehot = torch.eye(3)[device_onehot]

	device_onehot = move_data_to_gpu(device_onehot, True)

	x = self.emb(input, device_onehot)

        x = F.max_pool2d(x, kernel_size= (16, 16), stride=(16, 16))
        x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))

        output = F.log_softmax(self.fc_final(x), dim=-1)

        return output, device

class CnnAtrous_roi_attention(nn.Module):
    def __init__(self, classes_num, devices_num, cond_layer):
        super(CnnAtrous_roi_attention, self).__init__()

        self.emb = EmbeddingLayers_atrous(cond_layer)
        self.attention = Attention2d(
            512,
            classes_num,
            att_activation='sigmoid',
            cla_activation='log_softmax')

        self.cnn_device = BaselineCnn(devices_num)

    def init_weights(self):
        pass

    def forward(self, input):
        """(samples_num, feature_maps, time_steps, freq_num)"""
        device = self.cnn_device(input) 

        device_onehot = torch.argmax(device, dim=1)
	device_onehot = torch.eye(3)[device_onehot]

	device_onehot = move_data_to_gpu(device_onehot, True)

	x = self.emb(input, device_onehot)

        x = F.max_pool2d(x, kernel_size= (16, 16), stride=(16, 16))
        output = self.attention(x)

        return output, device

class CnnAtrous_Attention(nn.Module):
    def __init__(self, classes_num, devices_num, cond_layer):
        super(CnnAtrous_Attention, self).__init__()

        self.emb = EmbeddingLayers_atrous(cond_layer)
        self.attention = Attention2d(
            512,
            classes_num,
            att_activation='sigmoid',
            cla_activation='log_softmax')

        self.cnn_device = BaselineCnn(devices_num)

    def init_weights(self):
        pass

    def forward(self, input):
        """(samples_num, feature_maps, time_steps, freq_num)"""

        device = self.cnn_device(input) 

        device_onehot = torch.argmax(device, dim=1)
	device_onehot = torch.eye(3)[device_onehot]

	device_onehot = move_data_to_gpu(device_onehot, True)

        x = self.emb(input, device_onehot)

        output = self.attention(x)

        return output, device

#####################################################################################################
class EmbeddingLayers(nn.Module):
    def __init__(self, cond_layer=3):
        super(EmbeddingLayers, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64,
                               kernel_size=(5, 5), stride=(1, 1),
                               padding=(2, 2), bias=False)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128,
                               kernel_size=(5, 5), stride=(1, 1),
                               padding=(2, 2), bias=False)

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256,
                               kernel_size=(5, 5), stride=(1, 1),
                               padding=(2, 2), bias=False)

        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512,
                               kernel_size=(5, 5), stride=(1, 1),
                               padding=(2, 2), bias=False)
        
	self.condlayer = cond_layer
	if cond_layer==1:
            self.cond = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(1, 1))
	elif cond_layer==2:
            self.cond2 = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=(1, 1))
	elif cond_layer==3:
            self.cond3 = nn.Conv2d(in_channels=3, out_channels=256, kernel_size=(1, 1))
	elif cond_layer==4:
            self.cond4 = nn.Conv2d(in_channels=3, out_channels=512, kernel_size=(1, 1))

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)

        self.init_weights()

    def init_weights(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_layer(self.conv3)
        init_layer(self.conv4)

	if self.condlayer==1:
            init_layer(self.cond)
	elif self.condlayer==2:
            init_layer(self.cond2)
	elif self.condlayer==3:
            init_layer(self.cond3)
	elif self.condlayer==4:
            init_layer(self.cond4)

        init_bn(self.bn1)
        init_bn(self.bn2)
        init_bn(self.bn3)
        init_bn(self.bn4)

    def forward(self, input, device, return_layers=False):
        (batch_size, seq_len, mel_bins) = input.shape

        x = input.view(-1, 1, seq_len, mel_bins)
        """(samples_num, feature_maps, time_steps, freq_num)"""

	if self.condlayer==1:
            device1 = torch.unsqueeze(torch.unsqueeze(device, 2), 3)
            device1 = device1.expand(-1, -1, seq_len, mel_bins)

            x = F.relu(self.bn1(torch.add(self.conv1(x), self.cond(device1))))  # 1*1
            x = F.max_pool2d(x, kernel_size=(2, 2))
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.max_pool2d(x, kernel_size=(2, 2))
            x = F.relu(self.bn3(self.conv3(x)))
            x = F.max_pool2d(x, kernel_size=(2, 2))
            x = F.relu(self.bn4(self.conv4(x)))
            x = F.max_pool2d(x, kernel_size=(2, 2))

	elif self.condlayer==2:
            device2 = torch.unsqueeze(torch.unsqueeze(device, 2), 3)
            device2 = device2.expand(-1, -1, seq_len/2, mel_bins/2)

            x = F.relu(self.bn1(self.conv1(x)))
            x = F.max_pool2d(x, kernel_size=(2, 2))
            x = F.relu(self.bn2(torch.add(self.conv2(x), self.cond2(device2))))
            x = F.max_pool2d(x, kernel_size=(2, 2))
            x = F.relu(self.bn3(self.conv3(x)))
            x = F.max_pool2d(x, kernel_size=(2, 2))
            x = F.relu(self.bn4(self.conv4(x)))
            x = F.max_pool2d(x, kernel_size=(2, 2))

	elif self.condlayer==3:
            device3 = torch.unsqueeze(torch.unsqueeze(device, 2), 3)
            device3 = device3.expand(-1, -1, seq_len/4, mel_bins/4)

            x = F.relu(self.bn1(self.conv1(x)))
            x = F.max_pool2d(x, kernel_size=(2, 2))
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.max_pool2d(x, kernel_size=(2, 2))
	    x = F.relu(self.bn3(torch.add(self.conv3(x), self.cond3(device3))))
            x = F.max_pool2d(x, kernel_size=(2, 2))
            x = F.relu(self.bn4(self.conv4(x)))
            x = F.max_pool2d(x, kernel_size=(2, 2))

	elif self.condlayer==4:
            device4 = torch.unsqueeze(torch.unsqueeze(device, 2), 3)
            device4 = device4.expand(-1, -1, seq_len/8, mel_bins/8)

            x = F.relu(self.bn1(self.conv1(x)))
            x = F.max_pool2d(x, kernel_size=(2, 2))
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.max_pool2d(x, kernel_size=(2, 2))
            x = F.relu(self.bn3(self.conv3(x)))
            x = F.max_pool2d(x, kernel_size=(2, 2))
	    x = F.relu(self.bn4(torch.add(self.conv4(x), self.cond4(device4))))
            x = F.max_pool2d(x, kernel_size=(2, 2))

        if return_layers is False:
            return x
        else:
            return [x, x]

class DecisionLevelMaxPooling(nn.Module):
    def __init__(self, classes_num, devices_num, cond_layer):

        super(DecisionLevelMaxPooling, self).__init__()

        self.emb = EmbeddingLayers(cond_layer)
        self.fc_final = nn.Linear(512, classes_num)
        self.cnn_device = BaselineCnn(devices_num)

        self.init_weights()

    def init_weights(self):

        init_layer(self.fc_final)

    def forward(self, input):
        """input: (samples_num, channel, time_steps, freq_bins)
        """

        # (samples_num, channel, time_steps, freq_bins)
        device = self.cnn_device(input) 
        device_onehot = torch.argmax(device, dim=1)
	device_onehot = torch.eye(3)[device_onehot]
	device_onehot = move_data_to_gpu(device_onehot, True)

        x = self.emb(input, device_onehot)

        # (samples_num, 512, hidden_units)
        output = F.max_pool2d(x, kernel_size=x.shape[2:])
        output = output.view(output.shape[0:2])

        output = F.log_softmax(self.fc_final(output), dim=-1)

        return output, device

class DecisionLevelAvgPooling(nn.Module):
    def __init__(self, classes_num, devices_num, cond_layer):
        super(DecisionLevelAvgPooling, self).__init__()

        self.emb = EmbeddingLayers(cond_layer)
        self.fc_final = nn.Linear(512, classes_num)
        self.cnn_device = BaselineCnn(devices_num)

        self.init_weights()

    def init_weights(self):
        init_layer(self.fc_final)

    def forward(self, input):
        """input: (samples_num, channel, time_steps, freq_bins)
        """

        # (samples_num, channel, time_steps, freq_bins)
        device = self.cnn_device(input) 
        device_onehot = torch.argmax(device, dim=1)
	device_onehot = torch.eye(3)[device_onehot]
	device_onehot = move_data_to_gpu(device_onehot, True)

        x = self.emb(input, device_onehot)

        # (samples_num, 512, hidden_units)
        x = F.avg_pool2d(x, kernel_size=x.shape[2:])
        x = x.view(x.shape[0:2])

        output = F.log_softmax(self.fc_final(x), dim=-1)

        return output, device

class DecisionLevelFlatten(nn.Module):
    def __init__(self, classes_num, devices_num, cond_layer):
        super(DecisionLevelFlatten, self).__init__()

        self.emb = EmbeddingLayers(cond_layer)
        self.fc_final = nn.Linear(40960, classes_num)
        self.cnn_device = BaselineCnn(devices_num)

        self.init_weights()

    def init_weights(self):
        init_layer(self.fc_final)

    def forward(self, input):
        """input: (samples_num, channel, time_steps, freq_bins)
        """
        device = self.cnn_device(input) 

        device_onehot = torch.argmax(device, dim=1)
	device_onehot = torch.eye(3)[device_onehot]

	device_onehot = move_data_to_gpu(device_onehot, True)

        # (samples_num, channel, time_steps, freq_bins)
        x = self.emb(input, device_onehot)

        # (samples_num, 512, hidden_units)
        x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))

        output = F.log_softmax(self.fc_final(x), dim=-1)

        return output, device

class Attention2d(nn.Module):
    def __init__(self, n_in, n_out, att_activation, cla_activation):
        super(Attention2d, self).__init__()

        self.att_activation = att_activation
        self.cla_activation = cla_activation

        self.att = nn.Conv2d(
            in_channels=n_in, out_channels=n_out, kernel_size=(
                1, 1), stride=(
                1, 1), padding=(
                0, 0), bias=True)

        self.cla = nn.Conv2d(
            in_channels=n_in, out_channels=n_out, kernel_size=(
                1, 1), stride=(
                1, 1), padding=(
                0, 0), bias=True)

        self.init_weights()

    def init_weights(self):
        init_layer(self.att)
        init_layer(self.cla)
        self.att.weight.data.fill_(0.)

    def activate(self, x, activation):

        if activation == 'linear':
            return x

        elif activation == 'relu':
            return F.relu(x)

        elif activation == 'sigmoid':
            return F.sigmoid(x)+0.1

        elif activation == 'log_softmax':
            return F.log_softmax(x, dim=1)

    def forward(self, x):
        """input: (samples_num, channel, time_steps, freq_bins)
        """

        att = self.att(x)
        att = self.activate(att, self.att_activation)

        cla = self.cla(x)
        cla = self.activate(cla, self.cla_activation)

        # (samples_num, channel, time_steps * freq_bins)
        att = att.view(att.size(0), att.size(1), att.size(2) * att.size(3))
        cla = cla.view(cla.size(0), cla.size(1), cla.size(2) * cla.size(3))

        epsilon = 0.1 # 1e-7
        att = torch.clamp(att, epsilon, 1. - epsilon)

        norm_att = att / torch.sum(att, dim=2)[:, :, None]
        x = torch.sum(norm_att * cla, dim=2)

        Return_heatmap = False
        if Return_heatmap:
            return x, norm_att
        else:
            return x


class DecisionLevelSingleAttention(nn.Module):

    def __init__(self, classes_num, devices_num, cond_layer):

        super(DecisionLevelSingleAttention, self).__init__()

        self.emb = EmbeddingLayers(cond_layer)
        self.attention = Attention2d(
            512,
            classes_num,
            att_activation='sigmoid',
            cla_activation='log_softmax')
        self.cnn_device = BaselineCnn(devices_num)

    def init_weights(self):
        pass

    def forward(self, input):
        """input: (samples_num, freq_bins, time_steps, 1)
        """

        device = self.cnn_device(input)

        device_onehot = torch.argmax(device, dim=1)
	device_onehot = torch.eye(3)[device_onehot]

	device_onehot = move_data_to_gpu(device_onehot, True)

        # (samples_num, hidden_units, time_steps, 1)
        b1 = self.emb(input, device_onehot)

        # (samples_num, classes_num, time_steps, 1)
        output = self.attention(b1)

        return output, device
