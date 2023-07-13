import torch
import torch.nn as nn

class paraphraser(nn.Module):
	def __init__(self, in_channels_t, k, use_bn=True):
		super(paraphraser, self).__init__()
		factor_channels = int(in_channels_t*k)
		self.encoder = nn.Sequential(*[
				nn.Conv2d(in_channels_t, in_channels_t, 3, 1, 1, bias=bool(1-use_bn)),
				nn.BatchNorm2d(in_channels_t) if use_bn else nn.Sequential(),
				nn.LeakyReLU(0.1, inplace=True),
				nn.Conv2d(in_channels_t, factor_channels, 3, 1, 1, bias=bool(1-use_bn)),
				nn.BatchNorm2d(factor_channels) if use_bn else nn.Sequential(),
				nn.LeakyReLU(0.1, inplace=True),
				nn.Conv2d(factor_channels, factor_channels, 3, 1, 1, bias=bool(1-use_bn)),
				nn.BatchNorm2d(factor_channels) if use_bn else nn.Sequential(),
				nn.LeakyReLU(0.1, inplace=True),
			])
		self.decoder = nn.Sequential(*[
				nn.ConvTranspose2d(factor_channels, factor_channels, 3, 1, 1, bias=bool(1-use_bn)),
				nn.BatchNorm2d(factor_channels) if use_bn else nn.Sequential(),
				nn.LeakyReLU(0.1, inplace=True),
				nn.ConvTranspose2d(factor_channels, in_channels_t, 3, 1, 1, bias=bool(1-use_bn)),
				nn.BatchNorm2d(in_channels_t) if use_bn else nn.Sequential(),
				nn.LeakyReLU(0.1, inplace=True),
				nn.ConvTranspose2d(in_channels_t, in_channels_t, 3, 1, 1, bias=bool(1-use_bn)),
				nn.BatchNorm2d(in_channels_t) if use_bn else nn.Sequential(),
				nn.LeakyReLU(0.1, inplace=True),
			])

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
			if isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

	def forward(self, x):
		z   = self.encoder(x)
		out = self.decoder(z)
		return z, out

class translator(nn.Module):
	def __init__(self, in_channels_s, in_channels_t, k, use_bn=True):
		super(translator, self).__init__()
		factor_channels = int(in_channels_t*k)
		'''
		self.encoder = nn.Sequential(*[
				nn.Conv2d(in_channels_s, in_channels_s, 3, 1, 1, bias=bool(1-use_bn)),
				nn.BatchNorm2d(in_channels_s) if use_bn else nn.Sequential(),
				nn.LeakyReLU(0.1, inplace=True),
				nn.Conv2d(in_channels_s, factor_channels, 3, 1, 1, bias=bool(1-use_bn)),
				nn.BatchNorm2d(factor_channels) if use_bn else nn.Sequential(),
				nn.LeakyReLU(0.1, inplace=True),
				nn.Conv2d(factor_channels, factor_channels, 3, 1, 1, bias=bool(1-use_bn)),
				nn.BatchNorm2d(factor_channels) if use_bn else nn.Sequential(),
				nn.LeakyReLU(0.1, inplace=True),
			])
		'''
		self.encoder=nn.Sequential(*[
			nn.Conv2d(in_channels_s, in_channels_s, 1, bias=bool(1-use_bn)),
				nn.BatchNorm2d(in_channels_s) if use_bn else nn.Sequential(),
				nn.LeakyReLU(0.1, inplace=True),
				nn.Conv2d(in_channels_s, factor_channels, 1,  bias=bool(1-use_bn)),
				nn.BatchNorm2d(factor_channels) if use_bn else nn.Sequential(),
				nn.LeakyReLU(0.1, inplace=True),
				nn.Conv2d(factor_channels, factor_channels, 1, bias=bool(1-use_bn)),
				nn.BatchNorm2d(factor_channels) if use_bn else nn.Sequential(),
				nn.LeakyReLU(0.1, inplace=True),
		])

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
			if isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

	def forward(self, x):
		z   = self.encoder(x)
		return z

def transfer_cuda(nets,device):
	if isinstance(nets,list):
		res=[]
		for net in nets:
			#net = torch.nn.DataParallel(net).to(args.device)
			net = net.to(device)
			res.append(net)
	else:
			#nets = torch.nn.DataParallel(nets).to(args.device)
			nets = nets.to(device)
	return nets

class Adapter(nn.Module):
	def __init__(self,in_channels_s, in_channels_t):
		super(Adapter,self).__init__()
		self.encoder=nn.Sequential(*[
			nn.Conv2d(in_channels_s, in_channels_t, 1),
			nn.BatchNorm2d(in_channels_s),
			nn.LeakyReLU(0.1, inplace=True),
		])

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
			if isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

	def forward(self, x):
		z   = self.encoder(x)
		return z



def define_translator(in_channels_s, in_channels_t, k, device, use_bn=True):
	if isinstance(in_channels_s,int):
		net = translator(in_channels_s, in_channels_t, k, use_bn)
	else:
		net=[]
		for i in range(len(in_channels_s)):
			net.append(translator(in_channels_s[i],in_channels_t[i],k,use_bn))
	net=transfer_cuda(net,device)
	return net

def define_paraphraser(in_channels_t, k, device, use_bn=True):
	if isinstance(in_channels_t,int):
		net = paraphraser(in_channels_t, k, use_bn)
	else:
		net=[]
		for in_channels in in_channels_t:
			temp_paraphraser=paraphraser(in_channels, k, use_bn)
			net.append(temp_paraphraser)
	net=transfer_cuda(net,device)
	return net

class MIA(nn.Module):
	def __init__(self,in_channels_s, in_channels_t,factor,use_bn=True) -> None:
		super(MIA,self).__init__()
		factor_channels=in_channels_t
		self.factor=factor
		self.encoder = nn.Sequential(*[
			nn.Conv2d(in_channels_s, in_channels_s, 3, 1, 1, bias=bool(1-use_bn)),
			nn.BatchNorm2d(in_channels_s) if use_bn else nn.Sequential(),
			nn.LeakyReLU(0.1, inplace=True),
			nn.Conv2d(in_channels_s, factor_channels, 3, 1, 1, bias=bool(1-use_bn)),
			nn.BatchNorm2d(factor_channels) if use_bn else nn.Sequential(),
			nn.LeakyReLU(0.1, inplace=True),
			nn.Conv2d(factor_channels, factor_channels, 3, 1, 1, bias=bool(1-use_bn)),
			nn.BatchNorm2d(factor_channels) if use_bn else nn.Sequential(),
			nn.LeakyReLU(0.1, inplace=True),
		])
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
			if isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

	def forward(self, x):
		z   = self.encoder(x)
		z   = z * self.factor
		return z