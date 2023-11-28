# Implant Tooth Localization via 4D Diffusion Model
The PyTorch implementation based on open-AI Diffusion model

Related works

* [Guided-diffusion](https://github.com/openai/guided-diffusion) (NIPS 2021, github)
* [Make-A-Video](https://github.com/lucidrains/make-a-video-pytorch) (CVPR 2023, GitHub)

<video src="https://private-user-images.githubusercontent.com/26128046/286168752-8454f8f4-42f0-4968-8eab-44d6f4b11d77.mp4?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTEiLCJleHAiOjE3MDExNTc0NTEsIm5iZiI6MTcwMTE1NzE1MSwicGF0aCI6Ii8yNjEyODA0Ni8yODYxNjg3NTItODQ1NGY4ZjQtNDJmMC00OTY4LThlYWItNDRkNmY0YjExZDc3Lm1wND9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFJV05KWUFYNENTVkVINTNBJTJGMjAyMzExMjglMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjMxMTI4VDA3MzkxMVomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWVmYTY4NzgzYTc0NTJhYTUyMDZmZmY0ZmNhMTgwYTE5NDQ2YWE4ZDE1NWEwYjUzYzQ5MDM4ZjFkODhlOTQ2ZDMmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0.1ICYa5p5S-rN3SGcB0oGlZAZlgTOXpLDRrI1OHX_sLA"></video>



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
