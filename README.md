# Implant Tooth Localization via 4D Diffusion Model
The PyTorch implementation based on open-AI Diffusion model

Related works

* [Guided-diffusion](https://github.com/openai/guided-diffusion) (NIPS 2021, github)
* [Make-A-Video](https://github.com/lucidrains/make-a-video-pytorch) (CVPR 2023, GitHub)

<video src="https://private-user-images.githubusercontent.com/26128046/286167925-7dea2448-4f2c-4ebb-bef5-0dba5af3944b.mp4?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTEiLCJleHAiOjE3MDExNTc2NTcsIm5iZiI6MTcwMTE1NzM1NywicGF0aCI6Ii8yNjEyODA0Ni8yODYxNjc5MjUtN2RlYTI0NDgtNGYyYy00ZWJiLWJlZjUtMGRiYTVhZjM5NDRiLm1wND9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFJV05KWUFYNENTVkVINTNBJTJGMjAyMzExMjglMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjMxMTI4VDA3NDIzN1omWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWU5MmYwNjIyZjNhMTczOWI1Zjg0YTM2ZDUzZGVjNzYwNjE3ZDIxYTdhZTVhM2Q1YjJjMGQyOWQ0MjM5MGVmNzUmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0.-ya7I-SkIVQikptW9M7Bslxpcrtb9YDAqUz0TkRMf_o"></video>

<video src="https://private-user-images.githubusercontent.com/26128046/286168752-8454f8f4-42f0-4968-8eab-44d6f4b11d77.mp4?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTEiLCJleHAiOjE3MDExNTc2NTcsIm5iZiI6MTcwMTE1NzM1NywicGF0aCI6Ii8yNjEyODA0Ni8yODYxNjg3NTItODQ1NGY4ZjQtNDJmMC00OTY4LThlYWItNDRkNmY0YjExZDc3Lm1wND9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFJV05KWUFYNENTVkVINTNBJTJGMjAyMzExMjglMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjMxMTI4VDA3NDIzN1omWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTJjNGZmY2Y5NDA1MDY3MDg2YzIxMTIzNjAyNDdiZWY5YmYwYjk0MGVkOGFjYjk0Nzg5YzFiZWJmZWEzNWI2YjAmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0.eAVLy-YmIbawY90NXb1wBCrdyRk84fRnrxl0pyHjLt8"></video>

<video src="https://private-user-images.githubusercontent.com/26128046/286168810-b0be2ceb-0878-415d-b1cc-80ae0878e2b5.mp4?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTEiLCJleHAiOjE3MDExNTc2NTcsIm5iZiI6MTcwMTE1NzM1NywicGF0aCI6Ii8yNjEyODA0Ni8yODYxNjg4MTAtYjBiZTJjZWItMDg3OC00MTVkLWIxY2MtODBhZTA4NzhlMmI1Lm1wND9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFJV05KWUFYNENTVkVINTNBJTJGMjAyMzExMjglMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjMxMTI4VDA3NDIzN1omWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWNlNTdmMDlkZWVjMzc3ODlhNzJmODViMWM3MzRlYWY5ZTZhNDVjOTY1YTk5MDRhNmU1MTQ0MmRhMWIzNjM4NjkmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0.op9uJEsZ4bJ4EMZua08aB8U9KSRoibESC4OSieRavV4"></video>

<video src="https://private-user-images.githubusercontent.com/26128046/286169032-37d7d2cd-623b-45ad-922a-34d3b2bb9b84.mp4?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTEiLCJleHAiOjE3MDExNTc2NTcsIm5iZiI6MTcwMTE1NzM1NywicGF0aCI6Ii8yNjEyODA0Ni8yODYxNjkwMzItMzdkN2QyY2QtNjIzYi00NWFkLTkyMmEtMzRkM2IyYmI5Yjg0Lm1wND9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFJV05KWUFYNENTVkVINTNBJTJGMjAyMzExMjglMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjMxMTI4VDA3NDIzN1omWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTQ0MDRkMjlkMzU2ODE2MjExMDk5ZTA3NmNkMzk1MmM2MjRmNjJiMjc1YWY4OWM2MGViMDk0MTY3MmI1Y2NmNTUmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0.U0UcxzC5G9SILPa_gIcnYJoG-O8meY3ifk-wRRLB0zM"></video>



## Project Overview

This project utilizes Diffusion-based anomaly detection to localize implant areas in dental X-ray videos of patients. By processing 4D input/output in videos, we leveraged the spatial/temporal convolution and attention mechanisms from the make-a-video model (CVPR 23). This approach proved to be highly effective in localizing implant regions, especially in the absence of ground truth labels for these areas.

The entire framework has been implemented in PyTorch, showcasing its capability in handling complex video data and delivering precise anomaly detection in dental imaging. This innovative application of Diffusion models demonstrates a significant advancement in medical imaging analysis, providing a new avenue for dental diagnostics and treatment planning.



![reconstruction](https://private-user-images.githubusercontent.com/26128046/286171289-cd605e2f-403e-457a-a77e-fa52fda1615c.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTEiLCJleHAiOjE3MDExNTg2MzAsIm5iZiI6MTcwMTE1ODMzMCwicGF0aCI6Ii8yNjEyODA0Ni8yODYxNzEyODktY2Q2MDVlMmYtNDAzZS00NTdhLWE3N2UtZmE1MmZkYTE2MTVjLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFJV05KWUFYNENTVkVINTNBJTJGMjAyMzExMjglMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjMxMTI4VDA3NTg1MFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWYwYWZjNTcyODViY2Y4YWFmNGJiZTZmZDI2ZDgxYWE3NzlkYjNmZGM0OTUyMGU4NTNiNDhhMjYwNTgwNmY5MmImWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0.z08z49lEaBQWTCtJynrWgMaiDcahOH2YVFJoLydCux8)

Unlike the latest GAN-based anomaly detection model, f-AnoGAN (Medical Image  Analysis 54, 2019), our approach does not require an encoder and initiates the denoising process from $x_\lambda$ sample from  $q(x_\lambda | x_0)$. The accompanying figures demonstrate the comparative performance of localization with varying $\lambda$ values. This distinction highlights the efficiency and effectiveness of our method in anomaly detection, particularly in scenarios where traditional encoder-based models may not be suitable.

## Installation

#### Requirements

- Python 3.8
- PyTorch 2.1.0
- CUDA 11.6

#### Conda Installation

``` bash
# 1. Create a conda virtual environment.
conda create -n cad python=3.8
conda activate cad
pip install -r requirements.txt
```



## Dataset prepare

- Download `dvf_png.zip` from [A multimodal dental dataset facilitating machine learning research and clinic services](https://physionet.org/content/multimodal-dental-dataset/1.0.0/). 

- Unzip `dvf_png.zip` , and then rename the obtained folder as **cad** and put it under the `./datasets` directory. 

- Run 

  ```bash
  python prepare_dataset_pex.py
  ```
  
  
  

## Training 

This project supports multi-GPUs training. The following code shows an example for training the model with 128x128 images using 4 GPUs.

  ```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
--nproc_per_node=1 --master_port 15121 \
train.py --exp_name [EXP_NAME] --config config_xray.yaml

  ```

All configs for this experiment are saved in `./config/config_xray.yaml`. 
If you change the number of GPUs, you may need to modify the `batch_size` in `./config/config_xray.yaml.yaml` to ensure using a same `batch_size`.



## Inference

- Download the trained weights for [128x128 x-ray](https://drive.google.com/drive/u/1/folders/1SUwn_kctEviVKWMD2vLlY6kYu-Ckg99x) . Put the obtained checkpoints under `./checkpoints/x_ray_128` .

- Run the following code to evaluate the trained model:

  ```bash
  CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
  --nproc_per_node=1 --master_port 15121 \
  test.py --exp_name x_ray_128 --save_name x_ray_128 --sample_algorithm ddim --corrupt_level 600
  ```
