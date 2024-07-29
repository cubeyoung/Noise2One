from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--results_dir', type=str, default='/mnt/ssd1/ICCV/Imagedenoising/Results/', help='saves results here.')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        parser.add_argument('--valid_name', type=str, default='Kodak', help='noise style')
        parser.add_argument('--style', type=str, default='gauss25', help='noise style')
        parser.add_argument('--eval_style', type=str, default='gauss25', help='noise style')
        parser.add_argument('--backbone_name', type=str, default='backbone_S_pretrain_denoising_gaussian', help='models are saved here')
        parser.add_argument('--decoder_chekpoint', type=str, default='decoder_finetune_denoising_v1_gaussian_default_2', help='models are saved here')    
        parser.add_argument('--config', type=str, default='/home/user/research/ICCV2023/ICCV2023/Tune_diffusion_unet_MLP/configs/light.yaml', help='models are saved here')    
       # parser.add_argument('--dataroot', type=str, default='backbone_pretrain_denoising_v0', help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        parser.add_argument('--test_input', type=str, default='LD_LE', help='saves results here.')        
        # Dropout and Batchnorm has different behavioir during training and test.
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        parser.add_argument('--num_test', type=int, default=10000, help='how many test images to run')
        # rewrite devalue values
        parser.set_defaults(model='test')
        # To avoid cropping, the load_size should be the same as crop_size
        parser.set_defaults(load_size=parser.get_default('crop_size'))
        self.isTrain = False
        return parser
