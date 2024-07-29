import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
from torch.nn.utils import spectral_norm
import torch.nn.functional as F
###############################################################################
# Helper Functions
###############################################################################


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x): return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.5)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=5,T_mult=9, eta_min=1e-5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a generator

    Our current implementation provides two types of generators:
        U-Net: [unet_128] (for 128x128 input images) and [unet_256] (for 256x256 input images)
        The original U-Net paper: https://arxiv.org/abs/1505.04597

        Resnet-based generator: [resnet_6blocks] (with 6 Resnet blocks) and [resnet_9blocks] (with 9 Resnet blocks)
        Resnet-based generator consists of several Resnet blocks between a few downsampling/upsampling operations.
        We adapt Torch code from Justin Johnson's neural style transfer project (https://github.com/jcjohnson/fast-neural-style).


    The generator has been initialized by <init_net>. It uses RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'unet':
        net = UNet()   
    elif netG == 'dncnn':
        net = DnCNN()     
    elif netG == 'resnet':
        net = ResNet(input_nc, output_nc, 10, ngf, 1)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids)

import torch
import torch.nn as nn
from numpy.random import normal
from numpy.linalg import svd
from math import sqrt
import torch.nn.init
from .common import *
import math


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        # nn.init.uniform(m.weight.data, 1.0, 0.02)
        m.weight.data.normal_(mean=0, std=math.sqrt(2./9./64.)).clamp_(-0.025, 0.025)
        nn.init.constant(m.bias.data, 0.0)


class ResidualSequential(nn.Sequential):
    def __init__(self, *args):
        super(ResidualSequential, self).__init__(*args)

    def forward(self, x):
        out = super(ResidualSequential, self).forward(x)
        # print(x.size(), out.size())
        x_ = None
        if out.size(2) != x.size(2) or out.size(3) != x.size(3):
            diff2 = x.size(2) - out.size(2)
            diff3 = x.size(3) - out.size(3)
            # print(1)
            x_ = x[:, :, diff2 /2:out.size(2) + diff2 / 2, diff3 / 2:out.size(3) + diff3 / 2]
        else:
            x_ = x
        return out + x_

    def eval(self):
        print(2)
        for m in self.modules():
            m.eval()
        exit()


def get_block(num_channels, norm_layer, act_fun):
    layers = [
        nn.Conv2d(num_channels, num_channels, 3, 1, 1, bias=False),
        norm_layer(num_channels, affine=True),
        act(act_fun),
        nn.Conv2d(num_channels, num_channels, 3, 1, 1, bias=False),
        norm_layer(num_channels, affine=True),
    ]
    return layers

def conv(in_f, out_f, kernel_size, stride=1, bias=True, pad='zero', downsample_mode='stride'):
    downsampler = None
    if stride != 1 and downsample_mode != 'stride':

        if downsample_mode == 'avg':
            downsampler = nn.AvgPool2d(stride, stride)
        elif downsample_mode == 'max':
            downsampler = nn.MaxPool2d(stride, stride)
        elif downsample_mode  in ['lanczos2', 'lanczos3']:
            downsampler = Downsampler(n_planes=out_f, factor=stride, kernel_type=downsample_mode, phase=0.5, preserve_size=True)
        else:
            assert False

        stride = 1

    padder = None
    to_pad = int((kernel_size - 1) / 2)
    if pad == 'reflection':
        padder = nn.ReflectionPad2d(to_pad)
        to_pad = 0
  
    convolver = nn.Conv2d(in_f, out_f, kernel_size, stride, padding=to_pad, bias=bias)


    layers = filter(lambda x: x is not None, [padder, convolver, downsampler])
    return nn.Sequential(*layers)
def act(act_fun = 'LeakyReLU'):
    '''
        Either string defining an activation function or module (e.g. nn.ReLU)
    '''
    if isinstance(act_fun, str):
        if act_fun == 'LeakyReLU':
            return nn.LeakyReLU(0.2, inplace=True)
        elif act_fun == 'Swish':
            return Swish()
        elif act_fun == 'ELU':
            return nn.ELU()
        elif act_fun == 'none':
            return nn.Sequential()
        else:
            assert False
    else:
        return act_fun()


def bn(num_features):
    return nn.BatchNorm2d(num_features)

class ResNet(nn.Module):
    def __init__(self, num_input_channels, num_output_channels, num_blocks, num_channels, need_residual=False, act_fun='LeakyReLU', need_sigmoid=True, norm_layer=nn.BatchNorm2d, pad='zero'):
        '''
            pad = 'start|zero|replication'
        '''
        super(ResNet, self).__init__()

        if need_residual:
            s = ResidualSequential
        else:
            s = nn.Sequential

        stride = 1
        # First layers
        layers = [
            # nn.ReplicationPad2d(num_blocks * 2 * stride + 3),
            conv(num_input_channels, num_channels, 3, stride=1, bias=True, pad=pad),
            act(act_fun)
        ]
        # Residual blocks
        # layers_residual = []
        for i in range(num_blocks):
            layers += [s(*get_block(num_channels, norm_layer, act_fun))]
       
        layers += [
            nn.Conv2d(num_channels, num_channels, 3, 1, 1),
            norm_layer(num_channels, affine=True)
        ]

        layers += [
            conv(num_channels, num_output_channels, 3, 1, bias=True, pad=pad)
            
        ]
        self.model = nn.Sequential(*layers)   
    def add_noises(self,input,std):
        
        mu = torch.randn_like(input)
        
        return input + std*mu, mu        
    def forward(self, x,std):
        x_bar, mu = self.add_noises(x,std)
        log_prob = self.model(x_bar)
        loss = F.mse_loss(std*log_prob,-mu)
        return log_prob,loss

    def eval(self):
        self.model.eval()

    
class DnCNN(nn.Module):
    def __init__(self, num_layers=17, num_features=64):
        super(DnCNN, self).__init__()
        layers = [nn.Sequential(nn.Conv2d(1, num_features, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(inplace=True))]
        for i in range(num_layers - 2):
            layers.append(nn.Sequential(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
                                        nn.BatchNorm2d(num_features),
                                        nn.ReLU(inplace=True)))
        layers.append(nn.Conv2d(num_features, 1, kernel_size=1, padding=0))
        self.layers = nn.Sequential(*layers)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x_bar, mu = self.add_noises(x,std)
        residual = self.layers(x_bar)
        return residual
    
class UNet(nn.Module):
    """Custom U-Net architecture for Noise2Noise (see Appendix, Table 2)."""

    def __init__(self, in_channels=3, out_channels=3):
        """Initializes U-Net."""

        super(UNet, self).__init__()

        # Layers: enc_conv0, enc_conv1, pool1
        self._block1 = nn.Sequential(
            nn.Conv2d(in_channels*2, 48, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(48, 48, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2))

        # Layers: enc_conv(i), pool(i); i=2..5
        self._block2 = nn.Sequential(
            nn.Conv2d(48, 48, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2))

        # Layers: enc_conv6, upsample5
        self._block3 = nn.Sequential(
            nn.Conv2d(48, 48, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(48, 48, 3, stride=2, padding=1, output_padding=1))
            #nn.Upsample(scale_factor=2, mode='nearest'))

        # Layers: dec_conv5a, dec_conv5b, upsample4
        self._block4 = nn.Sequential(
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(96, 96, 3, stride=2, padding=1, output_padding=1))
            #nn.Upsample(scale_factor=2, mode='nearest'))

        # Layers: dec_deconv(i)a, dec_deconv(i)b, upsample(i-1); i=4..2
        self._block5 = nn.Sequential(
            nn.Conv2d(144, 96, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(96, 96, 3, stride=2, padding=1, output_padding=1))
            #nn.Upsample(scale_factor=2, mode='nearest'))

        # Layers: dec_conv1a, dec_conv1b, dec_conv1c,
        self._block6 = nn.Sequential(
            nn.Conv2d(96+(out_channels*2), 64, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(32, out_channels, 3, stride=1, padding=1))
        #self.weight = nn.Parameter(torch.ones([]) * 0.01)
        self.adapter = nn.Sequential(
            nn.Conv2d(in_channels, 48, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(48, in_channels, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.AdaptiveAvgPool2d((1,1)))        
        
        # Initialize weights
        self._init_weights()


    def _init_weights(self):
        """Initializes weights using He et al. (2015)."""

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()
                
    def add_noises(self,input,std):
        
        mu = torch.randn_like(input)
        
        return input + std*mu, mu 

    def forward(self, x,score):
        """Through encoder, then decoder by adding U-skip connections. """
        #x_bar, mu = self.add_noises(x,std)
        # Encoder
        #weights = self.weight.repeat(x.shape[0],1,1,1)
        weights = self.adapter(x)
        score = weights*score
        x = torch.cat([x,score],dim=1)
        pool1 = self._block1(x)
        pool2 = self._block2(pool1)
        pool3 = self._block2(pool2)
        pool4 = self._block2(pool3)
        pool5 = self._block2(pool4)

        # Decoder
        upsample5 = self._block3(pool5)
        concat5 = torch.cat((upsample5, pool4), dim=1)
        upsample4 = self._block4(concat5)
        concat4 = torch.cat((upsample4, pool3), dim=1)
        upsample3 = self._block5(concat4)
        concat3 = torch.cat((upsample3, pool2), dim=1)
        upsample2 = self._block5(concat3)
        concat2 = torch.cat((upsample2, pool1), dim=1)
        upsample1 = self._block5(concat2)
        concat1 = torch.cat((upsample1, x), dim=1)
        # Final activation
        log_prob = self._block6(concat1)
        #loss = F.mse_loss(std*log_prob,-mu)
        return log_prob #,loss

    def eval(self):
        self.model.eval()    
        
class UNet_Blind(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, depth=5, wf=48, slope=0.1):
        """
        Args:
            in_channels (int): number of input channels, Default 3
            depth (int): depth of the network, Default 5
            wf (int): number of filters in the first layer, Default 32
        """
        super(UNet_Blind, self).__init__()
        self.depth = depth
        self.head = nn.Sequential(
            LR(in_channels, wf, 3, slope), LR(wf, wf, 3, slope))
        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(LR(wf, wf, 3, slope))

        self.up_path = nn.ModuleList()
        for i in range(depth):
            if i != depth-1:
                self.up_path.append(UP(wf*2 if i==0 else wf*3, wf*2, slope))
            else:
                self.up_path.append(UP(wf*2+in_channels, wf*2, slope))

        self.last = nn.Sequential(LR(2*wf, 2*wf, 1, slope), 
                    LR(2*wf, 2*wf, 1, slope), conv1x1(2*wf, out_channels, bias=True))
                
    def add_noises(self,input,std):
        
        mu = torch.randn_like(input)
        
        return input + std*mu, mu 
    
    def forward(self, x, std):
        x_bar, mu = self.add_noises(x,std)
        blocks = []
        blocks.append(x_bar)
        x = self.head(x_bar)
        for i, down in enumerate(self.down_path):
            x = F.max_pool2d(x, 2)
            if i != len(self.down_path) - 1:
                blocks.append(x)
            x = down(x)

        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i-1])
            
        out = self.last(x)
        loss = F.mse_loss(std*out,-mu)
        return out, loss


class LR(nn.Module):
    def __init__(self, in_size, out_size, ksize=3, slope=0.1):
        super(LR, self).__init__()
        block = []
        block.append(nn.Conv2d(in_size, out_size,
                     kernel_size=ksize, padding=ksize//2, bias=True))
        block.append(nn.LeakyReLU(slope, inplace=True))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class UP(nn.Module):
    def __init__(self, in_size, out_size, slope=0.1):
        super(UP, self).__init__()
        self.conv_1 = LR(in_size, out_size)
        self.conv_2 = LR(out_size, out_size)

    def up(self, x):
        s = x.shape
        x = x.reshape(s[0], s[1], s[2], 1, s[3], 1)
        x = x.repeat(1, 1, 1, 2, 1, 2)
        x = x.reshape(s[0], s[1], s[2]*2, s[3]*2)
        return x

    def forward(self, x, pool):
        x = self.up(x)
        x = torch.cat([x, pool], 1)
        x = self.conv_1(x)
        x = self.conv_2(x)

        return x


def conv1x1(in_chn, out_chn, bias=True):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=1,
                      stride=1, padding=0, bias=bias)
    return layer      


class decoder(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, depth=3, wf=16, slope=0.1):
        super(decoder, self).__init__()
        self.depth = depth
        self.head = nn.Sequential(
            LR(in_channels*2, wf, 3, slope), LR(wf, wf, 3, slope))
        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(LR(wf, wf, 3, slope))

        self.up_path = nn.ModuleList()
        for i in range(depth):
            if i != depth-1:
                self.up_path.append(UP(wf*2 if i==0 else wf*3, wf*2, slope))
            else:
                self.up_path.append(UP(wf*2+(in_channels*2), wf*2, slope))

        self.last = nn.Sequential(LR(2*wf, 2*wf, 1, slope), 
                    LR(2*wf, 2*wf, 1, slope), conv1x1(2*wf, out_channels, bias=True))
        
        self.adapter = nn.Sequential(
            nn.Conv2d(in_channels, wf, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(wf, in_channels, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.AdaptiveAvgPool2d((1,1)))            
        
    def forward(self, x, score):

        weights = self.adapter(x)
        score = weights*score
        x = torch.cat([x,score],dim=1)             
        blocks = []
        blocks.append(x)
        x = self.head(x)
        for i, down in enumerate(self.down_path):
            x = F.max_pool2d(x, 2)
            if i != len(self.down_path) - 1:
                blocks.append(x)
            x = down(x)

        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i-1])            
        out = self.last(x)
        return out

class decoder_mlp(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, depth=3, wf=16, slope=0.1):
        super(decoder, self).__init__()
        self.depth = depth
        self.head = nn.Sequential(
            LR(in_channels*2, wf, 3, slope), LR(wf, wf, 3, slope))
        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(LR(wf, wf, 3, slope))

        self.up_path = nn.ModuleList()
        for i in range(depth):
            if i != depth-1:
                self.up_path.append(UP(wf*2 if i==0 else wf*3, wf*2, slope))
            else:
                self.up_path.append(UP(wf*2+(in_channels*2), wf*2, slope))

        self.last = nn.Sequential(LR(2*wf, 2*wf, 1, slope), 
                    LR(2*wf, 2*wf, 1, slope), conv1x1(2*wf, out_channels, bias=True))
        
        self.adapter = nn.Sequential(
            nn.Conv2d(in_channels, wf, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(wf, in_channels, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.AdaptiveAvgPool2d((1,1)))            
        
    def forward(self, x, score):
        weights = self.adapter(x)
        score = weights*score
        x = torch.cat([x,score],dim=1)             
        blocks = []
        blocks.append(x)
        x = self.head(x)
        for i, down in enumerate(self.down_path):
            x = F.max_pool2d(x, 2)
            if i != len(self.down_path) - 1:
                blocks.append(x)
            x = down(x)

        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i-1])            
        out = self.last(x)
        return out


                 
        