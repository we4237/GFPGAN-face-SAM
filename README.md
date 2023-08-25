RealESRNet
============================
https://github.com/xinntao/Real-ESRGAN

Requirement
----------------------------
* Argparse
* Numpy
* cv2
* Python 3.8
* PyTorch
* tqdm

Usage
----------------------------
### Testing
realesrnet_x1.pth\realesrnet_x2.pth  被在code.checkpoints里

Run the run.sh

```
$ python test_sr.py -h               

usage: generate_sample.py [--seed SEED] [--dataroot DATAROOT] [--output OUTPUT]
                          [--sr_scale SR_SCALE] 

optional arguments:
    --seed SEED          随机变量默认为1
    --dataroot DATAROOT  输入路径
    --output OUTPUT      输出路径
    --sr_scale SR_SCALE  选择 1 or 2,对应realesrnet_x1.pth\realesrnet_x2.pth 
                        是否需要将图像分辨率放大为4K
```
