# Implant Tooth Localization via 4D Diffusion Model
The PyTorch implementation based on open-AI Diffusion model

Related works

* [Guided-diffusion](https://github.com/openai/guided-diffusion) (NIPS 2021, github)
* [Make-A-Video](https://github.com/lucidrains/make-a-video-pytorch) (CVPR 2023, GitHub)



![sample](https://private-user-images.githubusercontent.com/26128046/286167925-7dea2448-4f2c-4ebb-bef5-0dba5af3944b.mp4?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTEiLCJleHAiOjE3MDExNTY5NjgsIm5iZiI6MTcwMTE1NjY2OCwicGF0aCI6Ii8yNjEyODA0Ni8yODYxNjc5MjUtN2RlYTI0NDgtNGYyYy00ZWJiLWJlZjUtMGRiYTVhZjM5NDRiLm1wND9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFJV05KWUFYNENTVkVINTNBJTJGMjAyMzExMjglMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjMxMTI4VDA3MzEwOFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTM4MmVmMTU5MDQ1NjVhMTg0ZmIzOGU3ZDZmYzgwMjllYjNkOWU2OTdlYmY1NDA1ZWZjMWI5OGQ2MjNhMDE3ZTAmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0.kJP_mKcLAwPpxnzK-gkKfsYDH0kCUzquySgYKPDn2Ws)





<video controls="" width="800" height="500" muted="" loop="" autoplay="">
<source src="https://private-user-images.githubusercontent.com/26128046/286167925-7dea2448-4f2c-4ebb-bef5-0dba5af3944b.mp4?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTEiLCJleHAiOjE3MDExNTY5NjgsIm5iZiI6MTcwMTE1NjY2OCwicGF0aCI6Ii8yNjEyODA0Ni8yODYxNjc5MjUtN2RlYTI0NDgtNGYyYy00ZWJiLWJlZjUtMGRiYTVhZjM5NDRiLm1wND9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFJV05KWUFYNENTVkVINTNBJTJGMjAyMzExMjglMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjMxMTI4VDA3MzEwOFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTM4MmVmMTU5MDQ1NjVhMTg0ZmIzOGU3ZDZmYzgwMjllYjNkOWU2OTdlYmY1NDA1ZWZjMWI5OGQ2MjNhMDE3ZTAmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0.kJP_mKcLAwPpxnzK-gkKfsYDH0kCUzquySgYKPDn2Ws" type="video/mp4">
</video>

This 



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
