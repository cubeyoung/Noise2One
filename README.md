# Noise2One
[ICASSP'24] Noise2One: One-Shot Image Denoising with Local Implicit Learning

Authors: Kwanyoung Kim, Jong Chul Ye   

---

## News
* [2024.07.29] Our official Code Release
* [2024.04.14] Our paper is accepted on ICASSP2024. 

## Environment:

- Install pytorch

 `conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio=0.10.1 cudatoolkit=10.2 -c pytorch`

- Install required packages.

 `pip install -r requirements.txt`

 ## Pretrained Score Model:

|     Dataset     |  Pretrained model zoo |
| :-------------: | :----------------------------------------------------------: |
| Gaussian  | [[Google Drive](https://drive.google.com/file/d/1UjyrFxQ0TTSDiQYse5MxAvHVyhJpqgMA/view?usp=drive_link)] |
| Poisson | [[Google Drive](https://drive.google.com/file/d/1SPXHxDl7znsOqUi9APnl3qx9BvmUbfny/view?usp=drive_link)] |

## Validation Datset:

|     Dataset     |  dataset download link |
| :-------------: | :----------------------------------------------------------: |
| CBSD300 | [[Google Drive](https://drive.google.com/file/d/1UjyrFxQ0TTSDiQYse5MxAvHVyhJpqgMA/view?usp=drive_link)] |
| Kodak24 | [[Google Drive](https://drive.google.com/file/d/1SPXHxDl7znsOqUi9APnl3qx9BvmUbfny/view?usp=drive_link)] |
| Set14   | [[Google Drive](https://drive.google.com/file/d/1SPXHxDl7znsOqUi9APnl3qx9BvmUbfny/view?usp=drive_link)] |

## Training (Inductive):
 ```shell
 `bash demo_kan.sh`
 ```

## Inference:
 `bash demo_kan.sh`

## Citation:
@inproceedings{kim2024noise2one,
  title={Noise2one: One-Shot Image Denoising with Local Implicit Learning},
  author={Kim, Kwanyoung and Ye, Jong Chul},
  booktitle={ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={13036--13040},
  year={2024},
  organization={IEEE}
}
