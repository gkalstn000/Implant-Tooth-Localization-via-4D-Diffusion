# Implant Tooth Localization via 4D Diffusion Model
The PyTorch implementation based on open-AI Diffusion model

Related works

* [Guided-diffusion](https://github.com/openai/guided-diffusion) (NIPS 2021, github)
* [Make-A-Video](https://github.com/lucidrains/make-a-video-pytorch) (CVPR 2023, GitHub)



<iframe src="https://private-user-images.githubusercontent.com/26128046/286163562-0ec2e8d6-01b3-4215-a52b-9a5bb3a33271.mp4?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTEiLCJleHAiOjE3MDExNTU4NjUsIm5iZiI6MTcwMTE1NTU2NSwicGF0aCI6Ii8yNjEyODA0Ni8yODYxNjM1NjItMGVjMmU4ZDYtMDFiMy00MjE1LWE1MmItOWE1YmIzYTMzMjcxLm1wND9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFJV05KWUFYNENTVkVINTNBJTJGMjAyMzExMjglMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjMxMTI4VDA3MTI0NVomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWQ4ZTgwYzM5ODZiOWVhOGVkMjBlYjNiYjcxOTczMThhODJmMTQ3NGRjMzE4NTQ2ZDQ1NDJlNWI2NzJjMzEzNjEmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0.WMC1VQBBmipCydRd-79Qz7nnZ5fdSB5k6R0XJE93CIE" frameborder="0" style="margin: 0 auto; display: block;" allowfullscreen></iframe>






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
