import torch
from .base_model import BaseModel
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from .ema import ExponentialMovingAverage
import torch.nn.functional as F
import os 
from .denoising_diffusion_pytorch2 import Unet
from .networks import UNet_Blind

class DiffusionModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
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
        self.visual_names = ['score','hr']
        if self.isTrain:
            self.model_names = ['netf']
        else:  # during test time, only load G
            self.model_names = ['netf']
            self.visual_names = ['lr','score','recon','hr']
        # define networks (both generator and discriminator)
        self.netf = UNet_Blind().to(self.device).to(self.device)
        #self.netf = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, 'unet', opt.norm,
        #                               not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        if self.isTrain:
            # define loss functions
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionL2 = torch.nn.MSELoss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_f = torch.optim.Adam(self.netf.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_f) 
        self.batch = opt.batch_size  
        self.sigma_min = 0.01
        self.sigma_max = 0.1
        self.sigma_annealing = 1000
        self.target_model = opt.target_model
        self.acc= 0
        self.sigmas = np.exp(np.linspace(np.log(self.sigma_max), np.log(self.sigma_min),self.sigma_annealing))
        self.sigmas = torch.from_numpy(self.sigmas)
        self.ema = ExponentialMovingAverage(self.netf.parameters(), decay=0.999)
        self.save_freq = 1      
        if self.isTrain:
            pass
        else:
            self.image_folder = opt.results_dir
            if not os.path.exists(self.image_folder):
                os.makedirs(self.image_folder)      
                
    def set_input(self, input):
        AtoB = self.opt.direction == 'AtoB'
        self.hr = input['hr'].to(self.device,dtype = torch.float32)
        self.image_paths = input['lr_paths']        

    def set_sigma(self, iter):
        labels = torch.randint(0, len(self.sigmas), (self.hr.shape[0],))
        self.sigma = self.sigmas[labels].view(self.hr.shape[0], *([1] * len(self.hr.shape[1:]))).to(self.device,dtype = torch.float32)
        self.loss_sigma = self.sigma[0]
        
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.ema.store(self.netf.parameters())
        self.ema.copy_to(self.netf.parameters())             
        self.zeros = torch.zeros(self.sigma.shape).to(self.device,dtype = torch.float32)                                 
        self.score = self.netf(self.hr,self.zeros)[0] 
        self.ema.restore(self.netf.parameters())
   
    def backward_f(self):
        _,self.loss_f = self.netf(self.hr,self.sigma)       
        self.loss_f.backward()
        
    def optimize_parameters(self):        
        self.optimizer_f.zero_grad()        # set G's gradients to zero
        self.backward_f()                   # calculate graidents for G
        torch.nn.utils.clip_grad_norm_(self.netf.parameters(), 1)        
        self.optimizer_f.step()              # udpate G's weights              
        self.ema.update(self.netf.parameters())
        with torch.no_grad():
            self.forward()
