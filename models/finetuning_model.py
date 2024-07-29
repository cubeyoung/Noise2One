import torch
from .base_model import BaseModel
import warnings
warnings.filterwarnings('ignore')
import yaml
import numpy as np
from .ema import ExponentialMovingAverage
import os 
from .networks import UNet_Blind, decoder
import lib as models
import cv2
from einops import rearrange
operation_seed_counter = 0

def get_generator():
    global operation_seed_counter
    operation_seed_counter += 1
    g_cuda_generator = torch.Generator(device="cuda")
    g_cuda_generator.manual_seed(operation_seed_counter)
    return g_cuda_generator    
def ssim(prediction, target):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2
    img1 = prediction.astype(np.float64)
    img2 = target.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) *
                (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                       (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(target, ref):
    '''
    calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    img1 = np.array(target, dtype=np.float64)
    img2 = np.array(ref, dtype=np.float64)
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1[:, :, i], img2[:, :, i]))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')
def calculate_psnr(target, ref, data_range=255.0):
    img1 = np.array(target, dtype=np.float32)
    img2 = np.array(ref, dtype=np.float32)
    diff = img1 - img2
    psnr = 10.0 * np.log10(data_range**2 / np.mean(np.square(diff)))
    return psnr   
def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret
class FinetuningModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(norm='batch', netG='unet', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=1.0, help='weight for L1 loss')

        return parser    
    
    def add_valid_noise(self, x):
        shape = x.shape
        if self.eval_style == "gauss_fix":
            std = self.eval_params[0]
            std = np.array(std)
            return np.array(x + np.random.normal(size=shape) * std,
                            dtype=np.float32), std
        elif self.eval_style == "gauss_range":
            min_std, max_std = self.eval_params
            std = np.random.uniform(low=min_std, high=max_std, size=(1, 1, 1))
            return np.array(x + np.random.normal(size=shape) * std,
                            dtype=np.float32), std      
        elif self.eval_style == "poisson_fix":
            lam = self.eval_params[0]
            return np.array(np.random.poisson(lam * x.numpy()) / lam, dtype=np.float32), np.array(lam)
        elif self.eval_style == "poisson_range":
            min_lam, max_lam = self.eval_params
            lam = np.random.uniform(low=min_lam, high=max_lam, size=(1, 1, 1))
            return np.array(np.random.poisson(lam * x.numpy()) / lam, dtype=np.float32), lam          
        elif self.eval_style == "gamma_fix":
            lam = self.eval_params[0]
            return np.array(x.numpy()*np.random.gamma(lam, 1/lam, shape), dtype=np.float32), np.array(lam)
        elif self.eval_style == "gamma_range":
            min_lam, max_lam = self.eval_params
            lam = np.random.uniform(low=min_lam, high=max_lam, size=(1, 1, 1))
            return np.array(x.numpy()*np.random.gamma(lam, 1/lam, shape), dtype=np.float32), lam 


    def add_train_noise(self, x):
        shape = x.shape
        if self.style == "gauss_fix":
            std = self.params[0]
            # x = x*0.5 + 0.5
            std = std * torch.ones((shape[0], 1, 1, 1), device=x.device)
            # noise = torch.cuda.FloatTensor(shape, device=x.device)
            noise = std * torch.randn_like(x, device = x.device)
            # torch.normal(mean=0.0,
            #              std=std,
            #              generator=get_generator(),
            #              out=noise)
            return x + noise , std
        elif self.style == "gauss_range":
            min_std, max_std = self.params
            std = torch.rand(size=(shape[0], 1, 1, 1),
                             device=x.device) * (max_std - min_std) + min_std
            noise = std * torch.randn_like(x, device = x.device)
            # torch.normal(mean=0, std=std, generator=get_generator(), out=noise)
            return x + noise , std
        elif self.style == "poisson_fix":
            lam = self.params[0]
            lam = lam * torch.ones((shape[0], 1, 1, 1), device=x.device)
            noised = torch.poisson(lam * x, generator=get_generator() ) / lam
            return noised, lam
        elif self.style == "poisson_range":
            min_lam, max_lam = self.params
            lam = torch.rand(size=(shape[0], 1, 1, 1),
                             device=x.device) * (max_lam - min_lam) + min_lam
            noised = torch.poisson(lam * x, generator=get_generator()) / lam
            return noised, lam
        elif self.style == "gamma_fix":
            lam = self.params[0]
            noised = torch.from_numpy(x.cpu().numpy()*np.random.gamma(lam, 1/lam, shape)).to(x.device) 
            lam = lam * torch.ones((shape[0], 1, 1, 1), device=x.device)
            return noised, lam

        elif self.style == "gamma_range":
            min_lam, max_lam = self.params
            lam = torch.rand(size=(shape[0], 1, 1, 1)) * (max_lam - min_lam) + min_lam
            
            noised = torch.from_numpy(x.cpu().numpy()*np.random.gamma(lam.numpy(), 1/lam.numpy(), shape)).to(x.device) 
            lam = lam.to(x.device)
            return noised, lam

    def load_pretrain(self, network, pretrain_dir,epoch = 'best', net = 'netf'):

        load_filename = '{}_net_{}.pth'.format(epoch, net) 
        load_path = os.path.join(pretrain_dir, load_filename)
        print('loading the model from %s' % load_path)
        state_dict = torch.load(load_path, map_location=self.device)
        network.load_state_dict(state_dict)  
        return network

    def load_ema(self, network, pretrain_dir,epoch = 'best', net = 'netf'):
        load_filename = '{}_ema_{}.pth'.format(epoch, net) 
        load_path = os.path.join(pretrain_dir, load_filename)
        print('loading the model from %s' % load_path)
        state_dict = torch.load(load_path, map_location=self.device)
        return state_dict

    # def load_pretrain_decoder(self, decoder, pretrain_dir):
    #     load_filename = 'best_net_decoder.pth' 
    #     load_path = os.path.join(pretrain_dir, load_filename)
    #     print('loading the model from %s' % load_path)
    #     state_dict = torch.load(load_path, map_location=self.device)
    #     decoder.load_state_dict(state_dict)  
    #     return decoder

    # def load_ema_decoder(self, decoder, pretrain_dir):
    #     load_filename = 'best_ema_decoder.pth' 
    #     load_path = os.path.join(pretrain_dir, load_filename)
    #     print('loading the model from %s' % load_path)
    #     state_dict = torch.load(load_path, map_location=self.device)
 
    #     return state_dict     
    
    # def load_pretrain_decoder(self, decoder, pretrain_dir):
    #     load_filename = 'best_net_decoder.pth' 
    #     load_path = os.path.join(pretrain_dir, load_filename)
    #     print('loading the model from %s' % load_path)
    #     state_dict = torch.load(load_path, map_location=self.device)
    #     decoder.load_state_dict(state_dict)  
    #     return decoder


    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['tuning'] #['tuning']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['input','lr','score','recon','hr']
        if self.isTrain:
            self.model_names = ['decoder']
            self.state_names = ['scheduler', 'optimizer']
        else:  # during test time, only load G
            self.model_names = ['decoder']
            self.visual_names = ['lr','recon','hr']
        with open(opt.config, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            print('config loaded.')   
        self.pretrain_dir = opt.checkpoints_dir +'/' + opt.backbone_name
        self.scorenet =  UNet_Blind().to(self.device)

        self.scorenet = self.load_pretrain(self.scorenet,self.pretrain_dir, epoch = 'best', net = "netf")
        self.loaded_state_score = self.load_ema(self.scorenet,self.pretrain_dir, epoch = 'best', net = "netf")
        for param in self.scorenet.parameters():
             param.requires_grad = False

        self.decoder = models.make(config['model']).to(self.device)  
        # self.decoder_dir = opt.checkpoints_dir +'/' + opt.decoder_chekpoint
        # self.decoder = self.load_pretrain(self.decoder,self.decoder_dir, epoch = 'best', net = "decoder")
        # self.loaded_state = self.load_ema(self.decoder,self.decoder_dir, epoch = 'best', net = "decoder")
        
        if not self.isTrain:
            self.decoder_dir = opt.checkpoints_dir +'/' + opt.decoder_chekpoint
            self.decoder = self.load_pretrain(self.decoder,self.decoder_dir, net = "decoder")
            self.loaded_state = self.load_ema(self.decoder,self.decoder_dir, net = "decoder")

        if self.isTrain:
            self.criterionL1 = torch.nn.SmoothL1Loss()
            self.criterionL2 = torch.nn.MSELoss()
            self.optimizer = torch.optim.AdamW(self.decoder.parameters(), lr=opt.lr, betas=(0.9, 0.999))
            self.optimizers.append(self.optimizer) 
        self.batch = opt.batch_size  
        self.ema_score = ExponentialMovingAverage(self.scorenet.parameters(), decay=0.999)
        self.ema = ExponentialMovingAverage(self.decoder.parameters(), decay=0.999)
        
        self.save_freq = 1      
        if self.isTrain:
            pass
        else:
            self.image_folder = opt.results_dir
            if not os.path.exists(self.image_folder):
                os.makedirs(self.image_folder)    
                  
        style = opt.style        
        print("Train Noise style", style)
        if style.startswith('gauss'):
            self.params = [
                float(p) / 255.0 for p in style.replace('gauss', '').split('_')
            ]
            if len(self.params) == 1:
                self.style = "gauss_fix"
            elif len(self.params) == 2:
                self.style = "gauss_range"     
                
        elif style.startswith('poisson'):
            self.params = [
                float(p) for p in style.replace('poisson', '').split('_')
            ]
            if len(self.params) == 1:
                self.style = "poisson_fix"
            elif len(self.params) == 2:
                self.style = "poisson_range"

        elif style.startswith('gamma'):
            self.params = [
                float(p) for p in style.replace('gamma', '').split('_')
            ]
            if len(self.params) == 1:
                self.style = "gamma_fix"
            elif len(self.params) == 2:
                self.style = "gamma_range"

        eval_style = opt.eval_style    
        print("Eval Noise style", eval_style)                
        if eval_style.startswith('gauss'):
            self.eval_params = [
                float(p) / 255.0 for p in eval_style.replace('gauss', '').split('_')
            ]
            if len(self.eval_params) == 1:
                self.eval_style = "gauss_fix"
            elif len(self.eval_params) == 2:
                self.eval_style = "gauss_range"     
                
        elif eval_style.startswith('poisson'):
            self.eval_params = [
                float(p) for p in eval_style.replace('poisson', '').split('_')
            ]
            if len(self.eval_params) == 1:
                self.eval_style = "poisson_fix"
            elif len(self.eval_params) == 2:
                self.eval_style = "poisson_range" 

        elif eval_style.startswith('gamma'):
            self.eval_params = [
                float(p) for p in eval_style.replace('gamma', '').split('_')
            ]
            if len(self.eval_params) == 1:
                self.eval_style = "gamma_fix"
            elif len(self.eval_params) == 2:
                self.eval_style = "gamma_range" 

                
    def set_input(self, input):
        self.hr = input['hr']
        self.gt = input['gt'].to(self.device,dtype = torch.float32)
        self.coord = input['coord'].to(self.device,dtype = torch.float32)
        self.cell = input['cell'].to(self.device,dtype = torch.float32)
        self.lr,self.std = self.add_train_noise(self.hr.to(self.device,dtype = torch.float32))
        self.image_paths = input['lr_paths']        
        self.hr = self.hr.to(self.device,dtype = torch.float32)
        self.lr = self.lr.to(self.device,dtype = torch.float32)
        
    def set_input_val(self, input):
        self.hr = input['hr']
        self.gt = input['gt']
        self.coord = input['coord'].to(self.device,dtype = torch.float32)
        self.cell = input['cell'].to(self.device,dtype = torch.float32)      
        self.lr, self.std = self.add_valid_noise(self.hr)
        self.std = torch.from_numpy(self.std).to(self.device,dtype = torch.float32)
        self.lr = torch.from_numpy(self.lr).to(self.device,dtype = torch.float32)
        self.image_paths = input['lr_paths']
        if len(self.hr.shape) ==3:
            self.hr = self.hr.unsqueeze(0)
            self.lr = self.lr.unsqueeze(0)
        self.origin255 = ((self.hr.permute(0,2,3,1)).cpu().data.clamp(0, 1).contiguous()*255).numpy().copy().squeeze(0)
        self.origin255 = self.origin255.astype(np.uint8) 

    def batched_predict(self, model, inp, score, coord, cell, bsize):
        with torch.no_grad():
            # model.gen_feat(inp,score)
            n = coord.shape[1]
            ql = 0
            preds = []
            while ql < n:
                qr = min(ql + bsize, n)
                pred = model.query_rgb(inp, score, coord[:, ql: qr, :], cell[:, ql: qr, :])
                # pred = model.query_rgb(coord[:, ql: qr, :], cell[:, ql: qr, :])
                preds.append(pred)
                ql = qr
            pred = torch.cat(preds, dim=1)
        return pred         
        
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        if self.isTrain:
            self.ema.store(self.decoder.parameters())
        else:
            self.ema.load_state_dict(self.loaded_state)
        self.ema.copy_to(self.decoder.parameters())             
        self.zeros =  torch.zeros(self.lr.shape[0],1,1,1).to(self.device,dtype = torch.float32)                               
        b,c,h,w = self.lr.shape                  
        self.score = self.scorenet(self.lr,self.zeros)[0]
        self.recon = self.batched_predict(self.decoder,self.lr,self.score, self.coord,self.cell, bsize = 30000)
        self.recon = self.recon.view(b,h,w,c).permute(0,3,1,2).cpu()
        self.ema.restore(self.decoder.parameters())    

    def forward_psnr(self,init):
        if init == True:
            self.ema_score.load_state_dict(self.loaded_state_score)
            self.ema_score.copy_to(self.scorenet.parameters())
            self.ema_score.copy_to(self.scorenet.parameters())

            self.ema.load_state_dict(self.loaded_state)
            self.ema.store(self.decoder.parameters())
            self.ema.copy_to(self.decoder.parameters())
        else:
            self.ema.store(self.decoder.parameters())
            self.ema.copy_to(self.decoder.parameters())

        self.scorenet.eval()
        self.decoder.eval()

        H = self.lr.shape[2]
        W = self.lr.shape[3]
        val_size = (max(H, W) + 31) // 32 * 32
        self.lr = np.pad(
            self.lr.squeeze(0).permute(1,2,0).cpu().numpy(),
         [[0, val_size - H], [0, val_size - W], [0, 0]],
        'reflect')
        self.lr = torch.from_numpy(self.lr).unsqueeze(0).permute(0,3,1,2,).to(self.device,dtype = torch.float32)   
        b,c,h,w = self.lr.shape                  
        coord = make_coord((h, w)).cuda()
        cell = torch.ones_like(coord)
        cell[:, 0] *= 2 / h
        cell[:, 1] *= 2 / w
        self.coord, self.cell = coord.unsqueeze(0).expand(b,-1,-1), cell.unsqueeze(0).expand(b,-1,-1)
        with torch.no_grad():
            self.zeros = torch.zeros(self.lr.shape[0],1,1,1).to(self.device,dtype = torch.float32)                               
            self.score = self.scorenet(self.lr,self.zeros)[0]  
            # self.recon = self.batched_predict(self.decoder,(self.lr -0.5) / 0.5,self.score, self.coord,self.cell, bsize = 30000)
            self.recon = self.decoder(self.lr,self.score,self.coord,self.cell)
            b,c,h,w = self.lr.shape
            self.recon = (self.recon).view(b,h,w,c).contiguous()
            # self.recon = rearrange(self.recon, 'b c h w -> b h w c')     
            self.recon = self.recon.cpu().data.clamp(0, 1).numpy().squeeze(0)
            self.recon = self.recon[:H, :W, :]
            pred255_dn = np.clip(self.recon * 255.0 + 0.5, 0,
                                255).astype(np.uint8)            
            psnr = calculate_psnr(self.origin255.astype(np.float32),pred255_dn.astype(np.float32))  
            ssim = calculate_ssim(self.origin255.astype(np.float32),pred255_dn.astype(np.float32))     
            self.ema.restore(self.decoder.parameters())


        self.lr = self.lr.permute(0,2,3,1).contiguous()
        self.lr = self.lr.cpu().data.clamp(0, 1).numpy().squeeze(0)
        self.lr = self.lr[:H, :W, :]

        self.hr = self.hr.permute(0,2,3,1).contiguous()
        self.hr = self.hr.cpu().data.clamp(0, 1).numpy().squeeze(0)
         
        return psnr,ssim
    
    def backward_decoder(self):
        
        batch_size, channel, height, width = self.lr.shape
        self.input = self.lr.contiguous().view(batch_size * height *width, channel) 
        self.input = self.input.contiguous().view(batch_size, channel, height, width)
        b,c,h,w = self.lr.shape
        with torch.no_grad():
            self.zeros = torch.zeros(self.lr.shape[0],1,1,1).to(self.device,dtype = torch.float32)                             
            self.score = self.scorenet(self.lr,self.zeros)[0]
        self.recon = self.decoder(self.lr, self.score,self.coord,self.cell)
        self.recon = torch.clamp(self.recon,0,1)
        self.loss_tuning = self.criterionL1(self.recon,self.gt)
        # self.recon= self.recon *0.5 + 0.5 
        # self.recon = self.recon1.contiguous().view(batch_size, channel, height, width)
        self.recon = self.recon.view(b,h,w,c).permute(0,3,1,2)
        self.loss = self.loss_tuning 
        self.loss.backward()
        
    def optimize_parameters(self):        
        self.optimizer.zero_grad()        
        self.backward_decoder()                   
        torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), 1)        
        self.optimizer.step()                       
        self.ema.update(self.decoder.parameters())
    
    def forward_test(self):
        self.zeros =  torch.zeros(self.lr.shape[0],1,1,1).to(self.device,dtype = torch.float32)                               
        self.score = self.scorenet(self.lr,self.zeros)[0]
        self.tweedie = self.lr + (self.std**2)*self.score
        self.tweedie = self.tweedie.permute(0,2,3,1).contiguous()
        self.tweedie = self.tweedie.cpu().data.clamp(0, 1).numpy().squeeze(0)

        pred255_dn = np.clip(self.recon * 255.0 + 0.5, 0,
                             255).astype(np.uint8)        
        pred255_tw = np.clip(self.tweedie * 255.0 + 0.5, 0,
                             255).astype(np.uint8)

        self.psnr = calculate_psnr(self.origin255.astype(np.float32),pred255_dn.astype(np.float32))      
        self.psnr_tw = calculate_psnr(self.origin255.astype(np.float32),pred255_tw.astype(np.float32))     
        self.ema.restore(self.decoder.parameters())    