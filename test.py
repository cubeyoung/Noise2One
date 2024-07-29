import os
from options.test_options import TestOptions
from data import create_dataset
from data.test_dataset import TestDataset
from models import create_model
from util.visualizer import save_images
import torchvision
from util import html
import torch
import numpy as np
import time 
if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   
    phase = 'test'
    # Validation Set
    Kodak_dir = os.path.join(opt.dataroot_valid, "Kodak24")
    BSD300_dir = os.path.join(opt.dataroot_valid, "BSD300/test")
    Set14_dir = os.path.join(opt.dataroot_valid, "Set14_ORIGN")
    Kodak_dataset = TestDataset(opt,Kodak_dir)
    BSD300_dataset = TestDataset(opt,BSD300_dir)
    Set14_dataset =  TestDataset(opt,Set14_dir)
    valid_dict = {
        "Kodak24": Kodak_dataset,
        "BSD300": BSD300_dataset,
        "Set14": Set14_dataset
    }   
    # dataset = create_dataset(opt,phase)  # create a dataset given opt.dataset_mode and other options    
    print('The number of test images = %d' % len(Kodak_dataset))
    print('The number of test images = %d' % len(BSD300_dataset))
    print('The number of test images = %d' % len(Set14_dataset))

    model = create_model(opt)      # create a model given opt.model and other options
     
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))

       
    model.eval()
    np.random.seed(11)   
    valid_repeat_times = {"Kodak24": 10, "BSD300": 3, "Set14": 20}
    # valid_repeat_times = {"Kodak24": 1, "BSD300": 1, "Set14": 1}
    for valid_name, valid_images in valid_dict.items():
        test_psnr = 0 
        cvpr_psnr = 0
        test_ssim = 0 
        cvpr_ssim = 0
    
        save_dir = "/mnt/ssd3/ICASSP2024/Noise2KAN/%s/Ours/%s/"%(opt.eval_style,valid_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)        
        repeat = valid_repeat_times[valid_name]
        start = time.time()
        for i in range(repeat):
            for i, data in enumerate(valid_images):
                model.set_input_val(data)
                with torch.no_grad():
                    psnr,ssim = model.forward_psnr(True)
                test_psnr += psnr
                test_ssim += ssim
                # cvpr_psnr += model.psnr_tw
                # cvpr_ssim += model.ssim_tw            
                # output = model.recon
                # output = torch.from_numpy(output).unsqueeze(0).permute(0,3,1,2)
                # img_path = model.get_image_paths()     
                # torchvision.utils.save_image(output, save_dir + '%s_%04d.jpg'%(valid_name,i+1))
        end = time.time()
        print(f"{end - start:.5f} sec")        
        test_psnr /= len(valid_images)*repeat     
        cvpr_psnr /= len(valid_images)*repeat  
        test_ssim /= len(valid_images)*repeat     
        cvpr_ssim /= len(valid_images)*repeat     
        # print('-----Average PSNR/SSIM for {} NA2Score----\n PSNR: {:.6f} dB; / SSIM: {:.6f}'.format(valid_name,cvpr_psnr, cvpr_ssim))
        print('-----Average PSNR/SSIM for {} Noise2One----\n PSNR: {:.6f} dB; / SSIM: {:.6f}'.format(valid_name,test_psnr, test_ssim))