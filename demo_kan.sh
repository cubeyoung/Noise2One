# ######## Tuning Gaussian 25 ####################
# python tuning.py --dataset_mode tuning --dataroot /mnt/ssd2/data0/CDIV2K_BSD_clean/4shot --name decoder_ft_gauss25_1shot_128_3layer_v1 \
#     --gpu_ids '1' --display_port 8750 --dataroot_valid /mnt/ssd3/Imagedenoising/Kodak24 --model finetuning --valid_name 'Kodak24'\
#     --backbone_name backbone_S_pretrain_gaussian25_55_v3 --style gauss25 --eval_style gauss25 --decoder_chekpoint decoder_ft_gauss25_1shot_128_3layer_v0  \

# ####### Tuning Gaussian 25 KAN ####################
# python tuning.py --dataset_mode tuning --dataroot /mnt/ssd2/data0/CDIV2K_BSD_clean/4shot --name decoder_ft_gauss25_1shot_4_3layer_kan \
#     --gpu_ids '1' --display_port 8750 --batch_size 8 --dataroot_valid /mnt/ssd3/Imagedenoising/Kodak24 --model finetuning --valid_name 'Kodak24'\
#     --backbone_name backbone_S_pretrain_gaussian25_55_v3 --style gauss25 --eval_style gauss25 --decoder_chekpoint decoder_ft_gauss25_1shot_128_3layer_v0  \

# ####### Tuning Gaussian 25 KAN_relu ####################
# python tuning.py --dataset_mode tuning --dataroot /mnt/ssd2/data0/CDIV2K_BSD_clean/4shot --config configs/light_relukan.yaml --name decoder_ft_gauss25_1shot_8_3layer_kan_relu \
#     --gpu_ids '1' --display_port 8850 --batch_size 8 --dataroot_valid /mnt/ssd3/Imagedenoising/Kodak24 --model finetuning --valid_name 'Kodak24'\
#     --backbone_name backbone_S_pretrain_gaussian25_55_v3 --style gauss25 --eval_style gauss25 \

# ######## Tuning Gaussian 25 MLP ####################
# python tuning.py --dataset_mode tuning --dataroot /mnt/ssd2/data0/CDIV2K_BSD_clean/4shot --config configs/light_mlp.yaml --name decoder_ft_gauss25_1shot_64_3layer_mlp \
#     --gpu_ids '1' --display_port 8750 --batch_size 2 --dataroot_valid /mnt/ssd3/Imagedenoising/Kodak24 --model finetuning --valid_name 'Kodak24'\
#     --backbone_name backbone_S_pretrain_gaussian25_55_v3 --style gauss25 --eval_style gauss25 --decoder_chekpoint decoder_ft_gauss25_1shot_128_3layer_v0  \

# ######## Test Gaussian 25  ####################
# python test.py --dataset_mode test --dataroot_valid /mnt/ssd3/Imagedenoising/ --config configs/light_kan.yaml --valid_name 'BSD300' \
#    --decoder_chekpoint decoder_ft_gauss25_1shot_4_3layer_kan  --backbone_name backbone_S_pretrain_gaussian25_55_v3 --eval_style gauss25 \
#    --name BSD300/decoder_ft_gauss25_1shot_128_3layer_v1 --gpu_ids '1' --model finetuning --epoch best --results_dir /mnt/ssd1/ICCV/Imagedenoising/Results/

# ######### Tuning Gaussian 50 ####################
# # python tuning.py --dataset_mode tuning --dataroot /mnt/ssd1/data0/CDIV2K_BSD_clean/4shot --name decoder_ft_gauss50_1shot_128_3layer_v2 \
# #     --gpu_ids '1' --display_port 8800 --dataroot_valid /mnt/ssd3/Imagedenoising/Kodak24 --model finetuning --valid_name 'Kodak24'\
# #     --backbone_name backbone_S_pretrain_gaussian25_55_v3 --style gauss50 --eval_style gauss50 --decoder_chekpoint decoder_ft_gauss50_1shot_128_3layer_v1 \

# ########## Test Gaussian 50  ####################
# python test.py --dataset_mode test --dataroot_valid /mnt/ssd3/Imagedenoising/ \
#    --decoder_chekpoint decoder_ft_gauss50_1shot_128_3layer_v2 --backbone_name backbone_S_pretrain_gaussian25_55_v3 --eval_style gauss50 \
#    --name BSD300/decoder_ft_gauss50_1shot_128_3layer_v2 --gpu_ids '0' --model finetuning --epoch best --results_dir /mnt/ssd1/ICCV/Imagedenoising/Results/


# ######### Poisson ####################
# ######### Poisson 30 kan ####################
# python tuning.py --dataset_mode tuning --dataroot /mnt/ssd2/data0/CDIV2K_BSD_clean/4shot --config configs/light_kan_16_16.yaml --name decoder_ft_poisson30_16_16_3layer_kan_v1 \
#    --gpu_ids '0' --display_port 8700 --dataroot_valid /mnt/ssd3/Imagedenoising/Kodak24 --model finetuning --valid_name 'Kodak24'\
#    --backbone_name backbone_S_pretrain_denoising_poisson25_55_v5 --style poisson30 --eval_style poisson30 --decoder_chekpoint decoder_ft_poisson30_128_3layer_v1 \

# python test.py --dataset_mode test --dataroot_valid /mnt/ssd3/Imagedenoising/ --config configs/light_kan_16_16.yaml \
#    --decoder_chekpoint decoder_ft_poisson30_16_16_3layer_kan_v1 --backbone_name backbone_S_pretrain_denoising_poisson25_55_v5 --eval_style poisson30 \
#    --name BSD300/decoder_ft_poisson30_16_16_3layer_kan_v1 --gpu_ids '0' --model finetuning --epoch best --results_dir /mnt/ssd1/ICCV/Imagedenoising/Results/

# # ######### Poisson 30 MLP(Noise2One) ####################
# python tuning.py --dataset_mode tuning --dataroot /mnt/ssd2/data0/CDIV2K_BSD_clean/4shot --config configs/light_mlp.yaml --name decoder_ft_poisson30_128_3layer_mlp_v0 \
#    --gpu_ids '1' --display_port 8710 --dataroot_valid /mnt/ssd3/Imagedenoising/Kodak24 --model finetuning --valid_name 'Kodak24'\
#    --backbone_name backbone_S_pretrain_denoising_poisson25_55_v5 --style poisson30 --eval_style poisson30 --decoder_chekpoint decoder_ft_poisson30_128_3layer_v1 \

# python tuning.py --dataset_mode tuning --dataroot /mnt/ssd2/data0/CDIV2K_BSD_clean/4shot --config configs/light_mlp.yaml --name decoder_ft_poisson30_light_128_3layer_mlp_v0 \
#    --gpu_ids '1' --display_port 8710 --dataroot_valid /mnt/ssd3/Imagedenoising/Kodak24 --model finetuning --valid_name 'Kodak24'\
#    --backbone_name backbone_S_pretrain_denoising_poisson25_55_v5 --style poisson30 --eval_style poisson30 --decoder_chekpoint decoder_ft_poisson30_128_3layer_v1 \

# python tuning.py --dataset_mode tuning --dataroot /mnt/ssd2/data0/CDIV2K_BSD_clean/4shot --config configs/light_mlp.yaml --name decoder_ft_poisson30_light_128_3layer_mlp_v0_coord \
#    --gpu_ids '1' --display_port 8710 --dataroot_valid /mnt/ssd3/Imagedenoising/Kodak24 --model finetuning --valid_name 'Kodak24'\
#    --backbone_name backbone_S_pretrain_denoising_poisson25_55_v5 --style poisson30 --eval_style poisson30 --decoder_chekpoint decoder_ft_poisson30_128_3layer_v1 \

# # ######### Poisson 30 Sine kan ####################
# python tuning.py --dataset_mode tuning --dataroot /mnt/ssd2/data0/CDIV2K_BSD_clean/4shot --config configs/light_sinekan_16_16.yaml --name decoder_ft_poisson30_16_16_3layer_sinekan_direct \
#    --gpu_ids '0' --display_port 8750 --dataroot_valid /mnt/ssd3/Imagedenoising/Kodak24 --model finetuning --valid_name 'Kodak24'\
#    --backbone_name backbone_S_pretrain_denoising_poisson25_55_v5 --style poisson30 --eval_style poisson30 --decoder_chekpoint decoder_ft_poisson30_128_3layer_v1 \

# # # ######### Poisson 30 vanila kan ####################
# python tuning.py --dataset_mode tuning --dataroot /mnt/ssd2/data0/CDIV2K_BSD_clean/4shot --config configs/light_kan_16_16.yaml --valid_epoch_freq 5 --name decoder_ft_poisson30_64_3layer_kan_direct \
#    --gpu_ids '0' --display_port 8750 --dataroot_valid /mnt/ssd3/Imagedenoising/Kodak24 --model finetuning --valid_name 'Kodak24'\
#    --backbone_name backbone_S_pretrain_denoising_poisson25_55_v5 --style poisson30 --eval_style poisson30 --decoder_chekpoint decoder_ft_poisson30_128_3layer_v1 \

# # # ######### Poisson 30 fast kan ####################
# python tuning.py --dataset_mode tuning --dataroot /mnt/ssd2/data0/CDIV2K_BSD_clean/4shot --config configs/light_fastkan_16_16.yaml --valid_epoch_freq 1 --name decoder_ft_poisson30_128_3layer_fastkan_liif_v2 \
#    --gpu_ids '0' --display_port 8750 --dataroot_valid /mnt/ssd3/Imagedenoising/Kodak24 --model finetuning --valid_name 'Kodak24'\
#    --backbone_name backbone_S_pretrain_denoising_poisson25_55_v5 --style poisson30 --eval_style poisson30 --decoder_chekpoint decoder_ft_poisson30_128_3layer_v1 \

# # # ######### Poisson 30 fast kan ####################
python tuning.py --dataset_mode tuning --dataroot /mnt/ssd2/data0/CDIV2K_BSD_clean/4shot --config configs/light_fastkan_128.yaml --valid_epoch_freq 1 --name decoder_ft_poisson30_128_3layer_fastkan_liif_v3 \
   --gpu_ids '1' --display_port 8850 --dataroot_valid /mnt/ssd3/Imagedenoising/Kodak24 --model finetuning --batch_size 4 --valid_name 'Kodak24'\
   --backbone_name backbone_S_pretrain_denoising_poisson25_55_v5 --style poisson30 --eval_style poisson30 --decoder_chekpoint decoder_ft_poisson30_128_3layer_v1 \


# # # # # ######### Poisson 30 fast kan ####################
# python tuning.py --dataset_mode tuning --dataroot /mnt/ssd2/data0/CDIV2K_BSD_clean/4shot --config configs/light_mlp.yaml --valid_epoch_freq 5 --name decoder_ft_poisson30_128_3layer_mlp_gt \
#    --gpu_ids '1' --display_port 8850 --dataroot_valid /mnt/ssd3/Imagedenoising/Kodak24 --batch_size 16 --model finetuningkan --valid_name 'Kodak24'\
#    --backbone_name backbone_S_pretrain_denoising_poisson25_55_v5 --style poisson30 --eval_style poisson30 --decoder_chekpoint decoder_ft_poisson30_128_3layer_v1 \

# # # # ######### Poisson 30 fast kan ####################
# python tuning.py --dataset_mode tuning --dataroot /mnt/ssd2/data0/CDIV2K_BSD_clean/4shot --config configs/light_fastkan_128.yaml --valid_epoch_freq 5 --name decoder_ft_poisson30_64_3layer_fastkan_gt \
#    --gpu_ids '0' --display_port 8750 --dataroot_valid /mnt/ssd3/Imagedenoising/Kodak24 --batch_size 16 --model finetuningkan --valid_name 'Kodak24'\
#    --backbone_name backbone_S_pretrain_denoising_poisson25_55_v5 --style poisson30 --eval_style poisson30 --decoder_chekpoint decoder_ft_poisson30_128_3layer_v1 \

# # ######### Poisson 30 Sine kan ####################
# python test.py --dataset_mode test --dataroot_valid /mnt/ssd3/Imagedenoising/ --config configs/light_sinekan_16_16.yaml \
#    --decoder_chekpoint decoder_ft_poisson30_16_16_3layer_sinekan_v0 --backbone_name backbone_S_pretrain_denoising_poisson25_55_v5 --eval_style poisson30 \
#    --name BSD300/decoder_ft_poisson30_16_16_3layer_sinekan_v0 --gpu_ids '0' --model finetuning --epoch best --results_dir /mnt/ssd1/ICCV/Imagedenoising/Results/

# # ######### Poisson 30  relukan ####################
# python tuning.py --dataset_mode tuning --dataroot /mnt/ssd2/data0/CDIV2K_BSD_clean/4shot --config configs/light_relukan.yaml --valid_epoch_freq 5 --name decoder_ft_poisson30_16_16_3layer_relukan_direct \
#    --gpu_ids '1' --display_port 8950 --dataroot_valid /mnt/ssd3/Imagedenoising/Kodak24 --model finetuning --valid_name 'Kodak24'\
#    --backbone_name backbone_S_pretrain_denoising_poisson25_55_v5 --style poisson30 --eval_style poisson30 --decoder_chekpoint decoder_ft_poisson30_128_3layer_v1 \

# ######### Poisson 50 ####################
# python tuning.py --dataset_mode tuning --dataroot /mnt/ssd1/data0/CDIV2K_BSD_clean/4shot --name decoder_ft_poisson50_128_3layer_v1 \
#    --gpu_ids '0' --display_port 8700 --dataroot_valid /mnt/ssd3/Imagedenoising/Kodak24 --model finetuning --config configs/light_poisson.yaml --valid_name 'Kodak24'\
#    --backbone_name backbone_S_pretrain_denoising_poisson25_55_v5 --style poisson50 --eval_style poisson50 --decoder_chekpoint decoder_ft_poisson50_128_3layer_v0 \
#    --estimator_chekpoint estimtaor_poisson25_55_v0

