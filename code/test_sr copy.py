import argparse
import os
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from torch2trt import torch2trt,TRTModule
from tqdm import tqdm

from utils import image, utils, metrics, parser
from time import time

from turbojpeg import TurboJPEG,TJPF_GRAY,TJSAMP_GRAY,TJFLAG_PROGRESSIVE,TJPF_BGR
from sr_model.rrdbnet_arch import RRDBNet
from data.ESRNet import esrNet

def load_srmodel(srmodel,config,device,base_dir='./'):
    loadnet = torch.load(os.path.join(base_dir, 'checkpoints', 'realesrnet_x%d.pth'%config.sr_scale))

    srmodel.load_state_dict(loadnet['params_ema'], strict=True)
    srmodel.eval()
    srmodel.half()
    srmodel = srmodel.to(device)


def uint2tensor3(img):
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float()
 

def test(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    srmodel = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=32, num_block=23, num_grow_ch=32, scale=config.sr_scale)

    load_srmodel(srmodel=srmodel,config=config,device = device ,base_dir=os.path.dirname(os.path.abspath(__file__)))

    # with torch.no_grad():
    #     data = torch.randn(1, 3, 1906, 1080).cuda().half()
    #     model_trt = torch2trt(srmodel, [data], fp16_mode=True)
    model_trt = TRTModule()
    model_trt.load_state_dict(torch.load('trt16.pth'))

    if os.path.isfile(config.dataroot):
        config.dataroot = [config.dataroot]
    # else:
    #     img_list = sorted(glob.glob(os.path.join(config.input, '*')))

    os.makedirs(args.output, exist_ok=True)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    test_loader = torch.utils.data.DataLoader(
        esrNet(config),
        batch_size=1,
        num_workers=1,
        pin_memory=True,
        shuffle=False, 
        drop_last=False
    )

    # jpeg = TurboJPEG() 
    if config.sr_scale == 2:
        mod_scale = 2
    elif config.sr_scale == 1:
        mod_scale = 4
    else:
        mod_scale = None
    with torch.no_grad():
        print("Testing...")

        for i,batch in tqdm(enumerate(test_loader)):
            start.record()

            lr_img = batch["lr"].to(device)


            # 当 sr_scale = 1 需要对图片进行pad
            if mod_scale is not None:
                h_pad, w_pad = 0, 0
                _, _, h, w = lr_img.size()
                if (h % mod_scale != 0):
                    h_pad = (mod_scale - h % mod_scale)
                if (w % mod_scale != 0):
                    w_pad = (mod_scale - w % mod_scale)
                lr_img = F.pad(lr_img, (0, w_pad, 0, h_pad), 'reflect') 

            out = model_trt(lr_img).squeeze() # C H W

            if mod_scale is not None:
                _, h, w = out.size()
                out = out[:, 0:h - h_pad, 0:w - w_pad]

            out *= 255.  

            # out = image.tensor2uint(out)
            out = out.round().clamp(0, 255).to(torch.uint8).transpose(0,2).transpose(0,1).cpu().numpy()
            out = out[:, :, [2, 1, 0]].cpu().numpy()
        
            # out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

            # 如果需要将x2的分辨率还原,通过resize还原
            # out = cv2.resize(out, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

            # 将图片输出到path
            path = f'{args.output}/{(i):05d}.png'
            
            # 使用TurboJPEG加速存储为jpg但是会降低一些画质
            # open_file = open(path,'wb')
            # open_file.write(jpeg.encode(out,quality=100))
            # open_file.close()

            cv2.imwrite(path, out)
            end.record()
            torch.cuda.synchronize()
            print(f'{start.elapsed_time(end)/1000:05f}' )


               

        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--dataroot", type=str, default='./dataroot')
    parser.add_argument("--output", type=str, default='./res')
    # data
    parser.add_argument('--sr_scale', type=int, default=2, help='SR scale')

    args = parser.parse_args()
    
    test(args)