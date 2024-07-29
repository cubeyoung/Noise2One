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
| Gaussian   | [[Google Drive](https://drive.google.com/drive/folders/11O8lSuHGdERBBDJ2R1QIYj6fEbAChCb6?usp=drive_link)] |
| Poisson | [[Google Drive](https://drive.google.com/drive/folders/1gKYwNia7WDZC1pt7yupXQHU6JzsK4omR?usp=drive_link)] |

## Validation Datset:

|     Dataset     |  dataset download link |
| :-------------: | :----------------------------------------------------------: |
| CBSD300 | [[Google Drive](https://drive.google.com/file/d/1jGMiw1JUHNAbA5ghkbpwfyaqP78B-k7z/view?usp=drive_link)] |
| Kodak24 | [[Google Drive](https://drive.google.com/file/d/1ZXE9zJ1F1Wk8MBN5pFP11R3En3Q56yRv/view?usp=drive_link)] |
| Set14   | [[Google Drive](https://drive.google.com/file/d/1RzTZXnPy-3A8oTCRWb-2_e7695qWZG7N/view?usp=drive_link)] |


## Few-Shot Datset:

|     Dataset     |  dataset download link |
| :-------------: | :----------------------------------------------------------: |
| One-Shot | [[Google Drive](https://drive.google.com/file/d/10ACiNmnlX6w-xlbEV8MSB2FIehLrsjFr/view?usp=sharing)] |


## Training (Inductive):
 ```shell
 `bash demo_kan.sh`
 ```

## Inference:
 `bash demo_kan.sh`

## Citation:
```
@inproceedings{kim2024noise2one,
  title={Noise2one: One-Shot Image Denoising with Local Implicit Learning},
  author={Kim, Kwanyoung and Ye, Jong Chul},
  booktitle={ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={13036--13040},
  year={2024},
  organization={IEEE}
}
```
