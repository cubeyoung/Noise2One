import torch
from .base_model import BaseModel
from . import networks
import torch.nn as nn
from skimage.measure import compare_psnr
import warnings
warnings.filterwarnings('ignore')
from util.util import calc_psnr
import numpy as np
from .mmd import mix_rbf_mmd2
import torch.nn.functional as F
from torch.autograd import Variable
from skimage.restoration import estimate_sigma   
from .radam import RAdam
class TestModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='unet', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=1.0, help='weight for L1 loss')

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['f','sigma']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['lr','recon']
        if self.isTrain:
            self.model_names = ['f']
        else:  # during test time, only load G
            self.model_names = ['f']
        # define networks (both generator and discriminator)
        self.netf = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, 'unet', opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        if self.isTrain:
            # define loss functions
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionL2 = torch.nn.MSELoss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_f = torch.optim.Adam(self.netf.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_f)
            
        self.variance = (opt.sigma/255)**2
        #self.variance = 
        self.batch = opt.batch_size  
        self.sigma_min = 0.01
        self.sigma_max = 25/255
        self.sigma_annealing = 500
        self.iter_g = 200
        self.iter_T = 5
        self.step = 3e-5
        self.sigmas = torch.tensor(
            np.exp(np.linspace(np.log(self.sigma_max), np.log(self.sigma_min),
                               self.iter_g))).float().to(self.device)
    def set_input(self, input):
        AtoB = self.opt.direction == 'AtoB'
        self.lr = input['B' if AtoB else 'A'].to(self.device,dtype = torch.float32)
        self.hr = input['B' if AtoB else 'A'].to(self.device,dtype = torch.float32)
        
        self.image_paths = input['B_paths' if AtoB else 'A_paths']
    def set_sigma(self, iter):
        perc = min((iter+1)/float(self.iter_g), 1.0)
        self.sigma = self.sigma_max * (1-perc) + self.sigma_min * perc
        self.loss_sigma = self.sigma
    def tweedie_eval(self,y_pred, y_true, p=1.5):
        a = y_true*torch.pow(y_pred, (1-p)) / (1-p)
        b = torch.pow(y_pred, (2-p))/(2-p)
        loss = torch.mean(-a + b)
        return loss         
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        with torch.no_grad():
            for i, sigma in enumerate(self.sigmas):
                sigma = sigma.item()
                self.alpha = self.step*(sigma/self.sigmas[-1])**2
                self.alpha = self.alpha.item()
                for t in range(self.iter_T):
                    self.n = torch.randn(self.lr.shape).to(self.device,dtype=torch.float32)
                    if i ==0:
                        self.score = self.netf(self.lr,sigma)[0]
                    else:
                        self.score = self.netf(self.x_n,sigma)[0] + (self.lr - self.x_n)/(self.variance - (sigma**2))
                    if i ==0:
                        self.x_n = self.lr + self.alpha*self.score + np.sqrt(2*self.alpha)*self.n
                    else:
                        self.x_n = self.x_n + self.alpha*self.score + np.sqrt(2*self.alpha)*self.n
        self.recon = self.x_n + (self.variance -(sigma**2))+self.netf(self.x_n,0)[0]
  
    def forward_psnr(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.recon = (self.variance)*self.netf(self.lr,0)[0]+self.lr
        self.recon = torch.clamp(self.recon.detach().cpu(), 0, 1)
        self.hr = self.hr.detach().cpu()
        psnr = calc_psnr(self.recon,self.hr)        
        return  psnr
    
    def backward_f(self):
        """Calculate GAN and L1 loss for the generator"""            
        self.output,self.loss_f = self.netf(self.lr,self.sigma)     
        self.loss_f.backward()
        
    def optimize_parameters(self):
        self.optimizer_f.zero_grad()        # set G's gradients to zero
        self.backward_f()                   # calculate graidents for G
        self.optimizer_f.step()              # udpate G's weights              