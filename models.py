import torch
import torch.nn as nn
import numpy as np
import math

class UpUint(nn.Module):
    def __init__(self,num_input_features,num_output_features,kernel_size,stride,padding):
    	super(UpUint,self).__init__()
    	self.deconv1 = nn.ConvTranspose2d(in_channels=num_input_features, out_channels=num_output_features,kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
    	self.relu1 = nn.PReLU()
    	self.conv1 = nn.Conv2d(in_channels=num_output_features, out_channels=num_input_features, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
    	self.relu2 = nn.PReLU()
    	self.deconv2 = nn.ConvTranspose2d(in_channels=num_input_features, out_channels=num_output_features,kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
    	self.relu3 = nn.PReLU()
    def forward(self,x):
    	h0 = self.relu1(self.deconv1(x))
    	l0 = self.relu2(self.conv1(h0))
    	diff = l0 - x
    	h1 = self.relu3(self.deconv2(diff))
    	out = h1 + h0
    	return out

class DownUint(nn.Module):
	def __init__(self,num_input_features,num_output_features,kernel_size,stride,padding):
		super(DownUint,self).__init__()
		self.conv1 = nn.Conv2d(in_channels=num_input_features, out_channels=num_output_features, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
		self.relu1 = nn.PReLU()
		self.deconv1 = nn.ConvTranspose2d(in_channels=num_output_features, out_channels=num_input_features,kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
		self.relu2 = nn.PReLU()
		self.conv2 = nn.Conv2d(in_channels=num_input_features, out_channels=num_output_features, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
		self.relu3 = nn.PReLU()
	def forward(self,x):
		l0 = self.relu1(self.conv1(x))
		h0 = self.relu2(self.deconv1(l0))
		diff = h0 - x
		l1 = self.relu3(self.conv2(diff))
		out = l1 + l0
		return out


class TranSition(nn.Module):
    def __init__(self,num_input_features,num_output_features):
    	super(TranSition,self).__init__()
    	self.conv1 = nn.Conv2d(in_channels=num_input_features, out_channels=num_output_features, kernel_size=1, stride=1, padding=0, bias=False)
    	self.relu1 = nn.PReLU()
    def forward(self,x):
    	out = self.relu1(self.conv1(x))
    	return out

class Net(nn.Module):
    def __init__(self,scale_factor):
    	super(Net,self).__init__()
    	if scale_factor == 2:
    		self.kernel_size = 6
    		self.stride = 2
    		self.padding = 2
    	elif scale_factor == 4:
    		self.kernel_size = 8
    		self.stride = 4
    		self.padding = 2
    	elif scale_factor == 8:
    		self.kernel_size = 12
    		self.stride = 8
    		self.padding = 2
    	self.conv1 = nn.Conv2d(3,256,kernel_size=3,stride=1,padding=1,bias=False)
    	self.relu1 = nn.PReLU()
    	self.conv2 = nn.Conv2d(256,64,kernel_size=1,stride=1,padding=0,bias=False)
    	self.relu2 = nn.PReLU()

    	self.up1 = UpUint(64,64,self.kernel_size,self.stride,self.padding)
    	self.down1 = DownUint(64,64,self.kernel_size,self.stride,self.padding)
        self.trans1 = TranSition(128,64)

    	self.up2 = UpUint(64,64,self.kernel_size,self.stride,self.padding)
    	self.trans2 = TranSition(128,64)
    	self.down2 = DownUint(64,64,self.kernel_size,self.stride,self.padding)
        self.trans22 = TranSition(192,64)

    	self.up3 = UpUint(64,64,self.kernel_size,self.stride,self.padding)
    	self.trans3 = TranSition(64*3,64)
    	self.down3 = DownUint(64,64,self.kernel_size,self.stride,self.padding)
        self.trans32 = TranSition(64*4,64)

    	self.up4 = UpUint(64,64,self.kernel_size,self.stride,self.padding)
    	self.trans4 = TranSition(64*4,64)
    	self.down4 = DownUint(64,64,self.kernel_size,self.stride,self.padding)
        self.trans42 = TranSition(64*5,64)

    	self.up5 = UpUint(64,64,self.kernel_size,self.stride,self.padding)
    	self.trans5 = TranSition(64*5,64)
    	self.down5 = DownUint(64,64,self.kernel_size,self.stride,self.padding)
        self.trans52 = TranSition(64*6,64)


    	self.up6 = UpUint(64,64,self.kernel_size,self.stride,self.padding)
    	self.trans6 = TranSition(64*6,64)
    	self.down6 = DownUint(64,64,self.kernel_size,self.stride,self.padding)
        self.trans62 = TranSition(64*7,64)

    	self.up7 = UpUint(64,64,self.kernel_size,self.stride,self.padding)
        self.trans71 = TranSition(64*7,64)

    	self.reconv = nn.Conv2d(64,3,kernel_size=3,stride=1,padding=1,bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                torch.nn.init.kaiming_normal(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self,x):
    	features = self.relu2(self.conv2(self.relu1(self.conv1(x))))
    	h1 = self.up1(features)
    	l1 = self.down1(h1)

        lin = torch.cat((features,l1),1)
        lin11 = self.trans1(lin)

        h2 = self.up2(lin11)
        hin = torch.cat((h1,h2),1)
        hin11 = self.trans2(hin)

        l2 = self.down2(hin11)
        lin2 = torch.cat((lin,l2),1)
        lin21 = self.trans22(lin2)

        h3 = self.up3(lin21)
        hin2 = torch.cat((hin,h3),1)
        hin21 = self.trans3(hin2)
        l3 = self.down3(hin21)

        lin3 = torch.cat((lin2,l3),1)
        lin31 = self.trans32(lin3)
        h4 = self.up4(lin31)
        hin3 = torch.cat((hin2,h4),1)
        hin31 = self.trans4(hin3)
        l4 = self.down4(hin31)

        lin4 = torch.cat((lin3,l4),1)
        lin41 = self.trans42(lin4)
        h5 = self.up5(lin41)

        hin4 = torch.cat((hin3,h5),1)
        hin41 = self.trans5(hin4)
        l5 = self.down5(hin41)
        lin5 = torch.cat((lin4,l5),1)
        lin51 = self.trans52(lin5)

        h6 = self.up6(lin51)
        hin5 = torch.cat((hin4,h6),1)
        hin51 = self.trans6(hin5)
        l6 = self.down6(hin51)
        lin6 = torch.cat((lin5,l6),1)
        lin61 = self.trans62(lin6)

        h7 = self.up7(lin61)

        hout = torch.cat((hin5,h7),1)
        hout = self.trans71(hout)

        out = self.reconv(hout)

        return out



class L1_Charbonnier_loss(nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt( diff * diff + self.eps )
        loss = torch.mean(error)
        return loss



