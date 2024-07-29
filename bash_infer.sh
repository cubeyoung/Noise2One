######### Tuning Gaussian 25 ####################
# python tuning.py --dataset_mode tuning --dataroot /mnt/ssd1/data0/CDIV2K_BSD_clean/4shot --name decoder_ft_gauss25_1shot_128_3layer_v1 \
#     --gpu_ids '1' --display_port 8750 --dataroot_valid /mnt/ssd3/Imagedenoising/Kodak24 --model finetuning --valid_name 'Kodak24'\
#     --backbone_name backbone_S_pretrain_gaussian25_55_v3 --style gauss25 --eval_style gauss25 --decoder_chekpoint decoder_ft_gauss25_1shot_128_3layer_v0  \

######### Test Gaussian 25  ####################
# python test.py --dataset_mode test --dataroot_valid /mnt/ssd3/Imagedenoising/ --valid_name 'BSD300' \
#    --decoder_chekpoint decoder_ft_gauss25_1shot_128_3layer_v1 --backbone_name backbone_S_pretrain_gaussian25_55_v3 --eval_style gauss25 \
#    --name BSD300/decoder_ft_gauss25_1shot_128_3layer_v1 --gpu_ids '0' --model finetuning --epoch best --results_dir /mnt/ssd1/ICCV/Imagedenoising/Results/

# ######### Tuning Gaussian 50 ####################
# # python tuning.py --dataset_mode tuning --dataroot /mnt/ssd1/data0/CDIV2K_BSD_clean/4shot --name decoder_ft_gauss50_1shot_128_3layer_v2 \
# #     --gpu_ids '1' --display_port 8800 --dataroot_valid /mnt/ssd3/Imagedenoising/Kodak24 --model finetuning --valid_name 'Kodak24'\
# #     --backbone_name backbone_S_pretrain_gaussian25_55_v3 --style gauss50 --eval_style gauss50 --decoder_chekpoint decoder_ft_gauss50_1shot_128_3layer_v1 \

# ########## Test Gaussian 50  ####################
python test.py --dataset_mode test --dataroot_valid /mnt/ssd3/Imagedenoising/ \
   --decoder_chekpoint decoder_ft_gauss50_1shot_128_3layer_v2 --backbone_name backbone_S_pretrain_gaussian25_55_v3 --eval_style gauss50 \
   --name BSD300/decoder_ft_gauss50_1shot_128_3layer_v3 --gpu_ids '0' --model finetuning --epoch best --results_dir /mnt/ssd1/ICCV/Imagedenoising/Results/

# ######### Gaussian 5_50 ####################
# python tuning.py --dataset_mode tuning --dataroot /mnt/ssd1/data0/CDIV2K_BSD_clean/4shot --name decoder_finetune_gauss50_128_3layer_v0\
#     --gpu_ids '1' --display_port 8500 --dataroot_valid /mnt/ssd3/Imagedenoising/Kodak24 --config configs/light.yaml --model finetuning --lr 1e-4 --valid_name 'Kodak' \
#     --backbone_name  backbone_S_pretrain_gaussian5_50 --style gauss50 --eval_style gauss50 --decoder_chekpoint decoder_finetune_gauss5_50_1shot_light_128_3layer_v1_new_backbone

#########Test Gaussian 5_50  ####################
# python test.py --dataset_mode test --dataroot_valid /mnt/ssd3/Imagedenoising/BSD300/test --valid_name 'BSD300' \
#    --decoder_chekpoint decoder_finetune_gauss5_50_1shot_light_128_3lalyer_v4 --backbone_name backbone_S_pretrain_denoising_gaussian --eval_style gauss5_50 \
#    --name BSD300/decoder_finetune_gauss5_50_1shot_light_128_3lalyer_v4 --gpu_ids '0' --model finetuning --epoch best --results_dir /mnt/ssd1/ICCV/Imagedenoising/Results/


# ######### Poisson 30 ####################
# python tuning.py --dataset_mode tuning --dataroot /mnt/ssd1/data0/CDIV2K_BSD_clean/4shot --name decoder_ft_poisson30_128_3layer_v2 \
#    --gpu_ids '0' --display_port 8600 --dataroot_valid /mnt/ssd3/Imagedenoising/Kodak24 --model finetuning --config configs/light_poisson.yaml --valid_name 'Kodak24'\
#    --backbone_name backbone_S_pretrain_denoising_poisson25_55_v5 --style poisson30 --eval_style poisson30 --decoder_chekpoint decoder_ft_poisson30_128_3layer_v1 \
#    --estimator_chekpoint estimtaor_poisson25_55_v0
# ######### Poisson 50 ####################
# python tuning.py --dataset_mode tuning --dataroot /mnt/ssd1/data0/CDIV2K_BSD_clean/4shot --name decoder_ft_poisson50_128_3layer_v1 \
#    --gpu_ids '0' --display_port 8700 --dataroot_valid /mnt/ssd3/Imagedenoising/Kodak24 --model finetuning --config configs/light_poisson.yaml --valid_name 'Kodak24'\
#    --backbone_name backbone_S_pretrain_denoising_poisson25_55_v5 --style poisson50 --eval_style poisson50 --decoder_chekpoint decoder_ft_poisson50_128_3layer_v0 \
#    --estimator_chekpoint estimtaor_poisson25_55_v0

######### Test Poisson 50  ####################
# python test.py --dataset_mode test --dataroot_valid /mnt/ssd3/Imagedenoising/ \
#    --decoder_chekpoint decoder_ft_poisson30_128_3layer_v2 --backbone_name backbone_S_pretrain_denoising_poisson25_55_v5 --eval_style poisson30 \
#    --name BSD300/decoder_ft_poisson30_128_3layer_v3 --gpu_ids '0' --model finetuning --epoch best --results_dir /mnt/ssd1/ICCV/Imagedenoising/Results/

# python test.py --dataset_mode test --dataroot_valid /mnt/ssd3/Imagedenoising/ \
#    --decoder_chekpoint decoder_ft_poisson50_128_3layer_v0 --backbone_name backbone_S_pretrain_denoising_poisson25_55_v5 --eval_style poisson50 \
#    --name BSD300/decoder_ft_poisson50_128_3layer_v0 --gpu_ids '0' --model finetuning --epoch best --results_dir /mnt/ssd1/ICCV/Imagedenoising/Results/


# ######### Gamma 25 ####################
# python tuning.py --dataset_mode tuning --dataroot /mnt/ssd1/data0/CDIV2K_BSD_clean/4shot --name decoder_ft_gamma25_128_3layer_v0 \
#    --gpu_ids '1' --display_port 8450 --dataroot_valid /mnt/ssd3/Imagedenoising/Kodak24 --model finetuning --config configs/light_poisson.yaml --valid_name 'Kodak24'\
#    --backbone_name backbone_S_pretrain_denoising_gamma25_55_v3 --style gamma25 --eval_style gamma25 --decoder_chekpoint decoder_ft_gamma25_128_3layer_v0 \
#    --estimator_chekpoint estimtaor_gamma25_55_v0
   
# ######### Gamma 50 ####################
# python tuning.py --dataset_mode tuning --dataroot /mnt/ssd1/data0/CDIV2K_BSD_clean/4shot --name decoder_ft_gamma50_128_3layer_v0 \
#    --gpu_ids '0' --display_port 8500 --dataroot_valid /mnt/ssd3/Imagedenoising/Kodak24 --model finetuning --config configs/light_poisson.yaml --valid_name 'Kodak24'\
#    --backbone_name backbone_S_pretrain_denoising_gamma25_55_v3 --style gamma50 --eval_style gamma50 --decoder_chekpoint decoder_ft_gamma50_128_3layer_v0 \
#    --estimator_chekpoint estimtaor_gamma25_55_v0

######### Test Poisson 50  ####################
# python test.py --dataset_mode test --dataroot_valid /mnt/ssd3/Imagedenoising/ \
#    --decoder_chekpoint decoder_ft_poisson30_128_3layer_v1 --backbone_name backbone_S_pretrain_denoising_poisson25_55_v5 --eval_style poisson30 \
#    --name BSD300/decoder_ft_poisson30_128_3layer_v1 --gpu_ids '0' --model finetuning --epoch best --results_dir /mnt/ssd1/ICCV/Imagedenoising/Results/

# python test.py --dataset_mode test --dataroot_valid /mnt/ssd3/Imagedenoising/ \
#    --decoder_chekpoint decoder_ft_poisson50_128_3layer_v0 --backbone_name backbone_S_pretrain_denoising_poisson25_55_v5 --eval_style poisson50 \
#    --name BSD300/decoder_ft_poisson50_128_3layer_v0 --gpu_ids '0' --model finetuning --epoch best --results_dir /mnt/ssd1/ICCV/Imagedenoising/Results/


# ######### Poisson 5_50 ####################
# python tuning.py --dataset_mode tuning --dataroot /mnt/ssd1/data0/CDIV2K_BSD_clean/4shot --name decoder_finetune_poisson5_50_1shot_128_3layer_v1 \
#     --gpu_ids '0' --display_port 8800 --dataroot_valid /mnt/ssd3/Imagedenoising/Kodak24 --config configs/blind_poisson.yaml --model finetuning --valid_name 'Kodak'\
#     --backbone_name backbone_S_pretrain_denoising_poisson5_50_v3 --style poisson5_50 --eval_style poisson5_50

# python tuning.py --dataset_mode tuning --dataroot /mnt/ssd1/data0/CDIV2K_BSD_clean/1shot --name decoder_finetune_poisson5_50_1shot_light_128_3lalyer \
#     --gpu_ids '0' --display_port 8800 --dataroot_valid /mnt/ssd3/Imagedenoising/BSD300/test --model finetuning \
#     --backbone_name  backbone_S_pretrain_denoising_poisson5_50 --style poisson5_50 --eval_style poisson5_50


#########Test Poisson 30  ####################
# python test.py --dataset_mode test --dataroot_valid /mnt/ssd3/Imagedenoising/BSD300/test --valid_name 'BSD300' \
#    --decoder_chekpoint decoder_finetune_poisson5_50_1shot_v4 --backbone_name backbone_S_pretrain_denoising_gaussian --eval_style poisson5_50 \
#    --name BSD300/decoder_finetune_poisson5_50_1shot_v4 --gpu_ids '0' --model finetuning --epoch best --results_dir /mnt/ssd1/ICCV/Imagedenoising/Results/

# python test.py --dataset_mode test --dataroot_valid /mnt/ssd3/Imagedenoising/Kodak --valid_name 'Kodak' \
#    --decoder_chekpoint decoder_finetune_poisson5_50_1shot_v4 --backbone_name  backbone_S_pretrain_denoising_gaussian --eval_style poisson5_50 \
#    --name Kodak/decoder_finetune_poisson5_50_1shot_v4 --gpu_ids '0' --model finetuning --epoch best --results_dir /mnt/ssd1/ICCV/Imagedenoising/Results/

# python test.py --dataset_mode test --dataroot_valid /mnt/ssd3/Imagedenoising/Set14 --valid_name 'Set14' \
#    --decoder_chekpoint decoder_finetune_poisson5_50_1shot_v4 --backbone_name  backbone_S_pretrain_denoising_gaussian --eval_style poisson5_50 \
#    --name Set14_C/decoder_finetune_poisson5_50_1shot_v4 --gpu_ids '0' --model finetuning --epoch best --results_dir /mnt/ssd1/ICCV/Imagedenoising/Results/






# # ########## Test Two-photon MICE  ####################

# python test.py --dataset_mode test --dataroot_valid /mnt/ssd3/Imagedenoising/microscope_data/TwoPhoton_MICE/ \
#    --decoder_chekpoint decoder_ft_gauss50_1shot_128_3layer_v2 --backbone_name backbone_S_pretrain_gaussian25_55_v3 --eval_style gauss50 \
#    --name BSD300/decoder_ft_gauss50_1shot_128_3layer_v3 --gpu_ids '0' --model finetuning --epoch best --results_dir /mnt/ssd1/ICCV/Imagedenoising/Results/








