import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from .models import register, make
from utils import make_coord
from models.networks import UNet_Blind
import numpy as np

class Sine(nn.Module):
    def __init__(self, w0 = 1.):
        super().__init__()
        self.w0 = w0
    def forward(self, x):
        return torch.sin(self.w0 * x)

@register('lte')
class LTE(nn.Module):
    def __init__(self, imnet_spec=None,
                 local_ensemble=True, feat_unfold=True, cell_decode=True, hidden_dim=64):
        super().__init__()
        self.local_ensemble = local_ensemble
        self.feat_unfold = feat_unfold
        self.cell_decode = cell_decode
        hidden_dim = imnet_spec['args']['hidden_list'][0]
        if imnet_spec is not None:
            imnet_in_dim = 3*2
            if self.feat_unfold:
                imnet_in_dim *= 9
            imnet_in_dim += 2 # attach coord
            if self.cell_decode:
                imnet_in_dim += 2
            self.imnet = make(imnet_spec, args={'in_dim': hidden_dim})
        else:
            self.imnet = None
        
        self.coef = nn.Conv2d(9, hidden_dim, 3, padding=1)
        self.freq = nn.Conv2d(9, hidden_dim, 3, padding=1)
        self.phase = nn.Linear(2, hidden_dim//2, bias=False)      
        
        self.adapter = nn.Sequential(
            nn.Conv2d(3, 128, 3, stride=1, padding=1),
            Sine(),
            nn.Conv2d(128, 3, 3, stride=1, padding=1),
            Sine())

        print(self.imnet)
        print(self.adapter)
        linear_params = sum(p.numel() for p in self.imnet.parameters())
        adapter_params = sum(p.numel() for p in self.adapter.parameters())

        print('[Linear decoder] Total number of parameters : %.3f K' % (linear_params / 1e3))
        print('[Noise Adapter] Total number of parameters : %.3f K' % (adapter_params / 1e3))

    def load_backbone(self, scorenet, pretrain_dir):
        load_filename = 'best_net_netf.pth' 
        load_path = os.path.join(pretrain_dir, load_filename)
        print('loading the model from %s' % load_path)
        state_dict = torch.load(load_path) #, map_location=self.device)
        scorenet.load_state_dict(state_dict)  
        return scorenet
        
    def gen_feat(self, inp):
        self.zeros = torch.zeros(inp.shape[0],1,1,1).to(inp.device,dtype = torch.float32) 
        self.score = self.scorenet(inp,self.zeros)[0]
        return self.score

    def query_rgb(self, input, score, coord, cell=None):

        self.input = input
        weights = self.adapter(input)
        input = torch.cat([input,score,weights],dim=1)  
        feat = input        

        coef = self.coef(feat)
        freq = self.freq(feat)   
        # if self.feat_unfold:
        #     feat = F.unfold(feat, 3, padding=1).view(
        #         feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3])

        # vx_lst = [-1, 1]
        # vy_lst = [-1, 1]
        # eps_shift = 1e-6
        vx_lst, vy_lst, eps_shift = [0], [0], 0
        # field radius (global: [-1, 1])
        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2

        feat_coord = make_coord(feat.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])

        preds = []
        areas = []
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                q_coef = F.grid_sample(
                    coef, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                q_freq = F.grid_sample(
                    freq, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                q_coord = F.grid_sample(
                    feat_coord, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]
                
                # prepare cell
                rel_cell = cell.clone()
                rel_cell[:, :, 0] *= feat.shape[-2]
                rel_cell[:, :, 1] *= feat.shape[-1]

                # basis generation
                bs, q = coord.shape[:2]
                q_freq = torch.stack(torch.split(q_freq, 2, dim=-1), dim=-1)
                q_freq = torch.mul(q_freq, rel_coord.unsqueeze(-1))
                q_freq = torch.sum(q_freq, dim=-2)
                q_freq += self.phase(rel_cell.view((bs * q, -1))).view(bs, q, -1)
                q_freq = torch.cat((torch.cos(np.pi*q_freq), torch.sin(np.pi*q_freq)), dim=-1)

                inp = torch.mul(q_coef, q_freq)            
                # inp = q_coef

                pred = self.imnet(inp.contiguous().view(bs * q, -1)).view(bs, q, -1)
                preds.append(pred)

                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)

        tot_area = torch.stack(areas).sum(dim=0)
        # t = areas[0]; areas[0] = areas[3]; areas[3] = t
        # t = areas[1]; areas[1] = areas[2]; areas[2] = t
        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(-1)
        ret += F.grid_sample(self.input, coord.flip(-1).unsqueeze(1), mode='bilinear',\
                padding_mode='border', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)            
        return ret

    def forward(self, inp, score, coord, cell):
        return self.query_rgb(inp, score, coord, cell)