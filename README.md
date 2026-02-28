# Lightweight modal-guided cross-attention fusion network for visible-infrared object detection
This is an official PyTorch implementation for our LCAFNet. Paper can be download in [LCAFNet](https://www.sciencedirect.com/science/article/pii/S0031320326003158)

### 1. Dependences
 Create a conda virtual environment and activate it.
 1) conda create --name MOD python=3.9
 2) conda activate MOD
 3) pip install -r requirements.txt

### 2. Datasets download
Download these datasets and create a dataset folder to hold them.
1) FLIR dataset: [FLIR](https://drive.google.com/file/d/1o9lchkdQcPaYqqEa_d_6l3QewyfkDTCx/view?usp=drive_link)
2) LLVIP dataset: [LLVIP](https://drive.google.com/file/d/1Bl1_D1T2x4JLu4__VbBjn6WJ3-T1Z99W/view?usp=drive_link)
3) M3FD dataset: [M3FD](https://drive.google.com/file/d/1FSfAQQ80UvwE7mXKDAxZZnabUrsM9HHD/view?usp=drive_link)
4) MFAD dataset: [MFAD](https://drive.google.com/file/d/1FSfAQQ80UvwE7mXKDAxZZnabUrsM9HHD/view?usp=drive_link)

### 3. Pretrained weights
Download our FreDFT weights and create a weights folder to hold them.
1) FLIR dataset: [LCAFNet_FLIR.pt](https://drive.google.com/file/d/1pIr9cFdbXpeLAhanBMoLqN48tWtdUjmK/view?usp=drive_link)
2) LLVIP dataset: [LCAFNet_LLVIP.pt](https://drive.google.com/file/d/1NuHgIarBmKYPACKKTY5F7AuhZsp08m5B/view?usp=drive_link)
3) M3FD dataset: [LCAFNet_M3FD.pt](https://drive.google.com/file/d/1Z90zNNTDbBosfVDLqqlrJY3YQvUS-Qu2/view?usp=drive_link)
4) MFAD dataset: [LCAFNet_MFAD.pt](https://drive.google.com/file/d/1Z90zNNTDbBosfVDLqqlrJY3YQvUS-Qu2/view?usp=drive_link)

### 4. Training our LCAFNet
Dataset path, GPU, batch size, etc., need to be modified according to different situations.
```
python train.py
```

### 5. Test our LCAFNet

```
python test.py
```

### 6. Citation
If you find LCAFNet helpful for your research, please consider citing our work.
```BibTex
@article{Wu2026,
  author       = {Wencong Wu and
                  Hongxi Zhang and
                  Xiuwei Zhang and
                  Hanlin Yin and
                  Yanning Zhang},
  title        = {Lightweight modal-guided cross-attention fusion network for visible-infrared object detection},
  journal      = {Pattern Recognition},
  volume       = {177},
  pages        = {113350},
  year         = {2026}
}
```

