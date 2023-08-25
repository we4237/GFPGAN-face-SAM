import argparse
import os
import cv2
import torch
import numpy as np
import torch.nn.functional as F
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from tqdm import tqdm

from utils import image, utils
from time import time

# from turbojpeg import TurboJPEG,TJPF_GRAY,TJSAMP_GRAY,TJFLAG_PROGRESSIVE,TJPF_BGR
from sr_model.rrdbnet_arch import RRDBNet
from data.ESRNet import esrNet

def load_srmodel(srmodel,config,device,base_dir='./'):
    loadnet = torch.load(os.path.join(base_dir, 'checkpoints', 'realesrnet_x%d.pth'%config.sr_scale))

    srmodel.load_state_dict(loadnet['params_ema'], strict=True)
    srmodel.eval()
    srmodel = srmodel.to(device)

def uint2tensor3(img):
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float()
 
# 导出onnx
def pre(config):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    srmodel = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=32, num_block=23, num_grow_ch=32, scale=config.sr_scale)
    load_srmodel(srmodel=srmodel,config=config,device=device,base_dir=os.path.dirname(os.path.abspath(__file__)))

    test_loader = torch.utils.data.DataLoader(
            esrNet(config),
            batch_size=1,
            num_workers=1,
            pin_memory=True,
            shuffle=False, 
            drop_last=False
        )
    with torch.no_grad():
        b,c,h,w = next(iter(test_loader))['lr'].shape
        x = torch.randn(b,c,h,w).to(device)
        
        torch.onnx.export(srmodel,
                        x,
                        '1.onnx',
                        opset_version=11,
                        input_names=["image"], 
                        output_names=["output"])

    #之后使用trtexec将onnx转为trt格式 ./trtexec --onnx= --saveEngine= --explicitBatch --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw --fp16 --device=3 
    #已将batchsize为1的trt打包在checkpoints里,直接运行即可

def postprocess(out):
    # out = torch.from_numpy(out)
    out *= 255.
    out = out.round().clamp(0, 255).to(torch.uint8).transpose(0,2).transpose(0,1).cpu().numpy()
    out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    return out

def trt_version():
    return trt.__version__

def torch_dtype_from_trt(dtype):
    if dtype == trt.int8:
        return torch.int8
    elif trt_version() >= '7.0' and dtype == trt.bool:
        return torch.bool
    elif dtype == trt.int32:
        return torch.int32
    elif dtype == trt.float16:
        return torch.float16
    elif dtype == trt.float32:
        return torch.float32
    else:
        raise TypeError("%s is not supported by torch" % dtype)
    
def torch_device_from_trt(device):
    if device == trt.TensorLocation.DEVICE:
        return torch.device("cuda")
    elif device == trt.TensorLocation.HOST:
        return torch.device("cpu")
    else:
        return TypeError("%s is not supported by torch" % device)   
     
class TRTModule(torch.nn.Module):
    def __init__(self, engine=None, input_names=None, output_names=None):
        super(TRTModule, self).__init__()
        self.engine = engine
        if self.engine is not None:
            # engine创建执行context
            self.context = self.engine.create_execution_context()

        self.input_names = input_names
        self.output_names = output_names

    def forward(self, *inputs):
        batch_size = inputs[0].shape[0]
        bindings = [None] * (len(self.input_names) + len(self.output_names))

        for i, input_name in enumerate(self.input_names):
            idx = self.engine.get_binding_index(input_name)
            # 设定shape 
            self.context.set_binding_shape(idx, tuple(inputs[i].shape))
            bindings[idx] = inputs[i].contiguous().data_ptr()

        # create output tensors
        outputs = [None] * len(self.output_names)
        for i, output_name in enumerate(self.output_names):
            idx = self.engine.get_binding_index(output_name)
            dtype = torch_dtype_from_trt(self.engine.get_binding_dtype(idx))
            shape = tuple(self.context.get_binding_shape(idx))
            device = torch_device_from_trt(self.engine.get_location(idx))
            output = torch.empty(size=shape, dtype=dtype, device=device)
            outputs[i] = output
            bindings[idx] = output.data_ptr()

        stream = torch.cuda.current_stream().cuda_stream

        self.context.execute_async_v2(bindings,
                                      stream)

        outputs = tuple(outputs)
        if len(outputs) == 1:
            outputs = outputs[0]

        return outputs
         
def test(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    srmodel = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=32, num_block=23, num_grow_ch=32, scale=config.sr_scale)
    load_srmodel(srmodel=srmodel,config=config,device = device ,base_dir=os.path.dirname(os.path.abspath(__file__)))

    if os.path.isfile(config.dataroot):
        config.dataroot = [config.dataroot]

    os.makedirs(args.output, exist_ok=True)
     
    # BATCH_SIZE = 1          
    # USE_FP16 = True                                      
    # target_dtype = np.float16 if USE_FP16 else np.float32   

    # f = open("code/checkpoints/test16.trt", "rb")           # 读取trt模型
    # runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))   # 创建一个Runtime(传入记录器Logger)
    # engine = runtime.deserialize_cuda_engine(f.read())      # 从文件中加载trt引擎
    # context = engine.create_execution_context()             # 创建context

    # # 4. 分配input和output内存
    # input_batch = np.random.randn(1, 3, 1906, 1080).astype(target_dtype)
    # output = np.empty([1, 3,3812,2160], dtype = target_dtype)

    # d_input = cuda.mem_alloc(1 * input_batch.nbytes)
    # d_output = cuda.mem_alloc(1 * output.nbytes)

    # bindings = [int(d_input), int(d_output)]

    # stream = cuda.Stream()

    # def predict(batch): # result gets copied into output
    #     # transfer input data to device
    #     cuda.memcpy_htod_async(d_input, batch, stream)
    #     # execute model
    #     context.execute_async_v2(bindings, stream.handle, None)  # 此处采用异步推理。如果想要同步推理，需将execute_async_v2替换成execute_v2
    #     # transfer predictions back
    #     cuda.memcpy_dtoh_async(output, d_output, stream)
    #     # syncronize threads
    #     stream.synchronize()
    #     return output

    # 0711
    logger = trt.Logger(trt.Logger.INFO)
    with open("code/checkpoints/test16.trt", "rb") as f, trt.Runtime(logger) as runtime:
        engine=runtime.deserialize_cuda_engine(f.read())

    for idx in range(engine.num_bindings):
        is_input = engine.binding_is_input(idx)
        name = engine.get_binding_name(idx)
        op_type = engine.get_binding_dtype(idx)
        # model_all_names.append(name)
        shape = engine.get_binding_shape(idx)

        print('input id:',idx,'   is input: ', is_input,'  binding name:', name, '  shape:', shape, 'type: ', op_type)
    # engine即上述加载的engine
    trt_model = TRTModule(engine, input_names=["input"],output_names=["output"])

    test_loader = torch.utils.data.DataLoader(
        esrNet(config),
        batch_size=1,
        num_workers=1,
        pin_memory=True,
        shuffle=False, 
        drop_last=False
    )
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    if config.sr_scale == 2:
        mod_scale = 2
    elif config.sr_scale == 1:
        mod_scale = 4
    else:
        mod_scale = None
    with torch.no_grad():
        print("Testing...")
        start.record()
        for i,batch in tqdm(enumerate(test_loader)):
            
            # lr_img = batch["lr"].to(device)
            lr_img = batch["lr"]

            # 当 sr_scale = 1 需要对图片进行pad
            if mod_scale is not None:
                h_pad, w_pad = 0, 0
                _, _, h, w = lr_img.shape
                if (h % mod_scale != 0):
                    h_pad = (mod_scale - h % mod_scale)
                if (w % mod_scale != 0):
                    w_pad = (mod_scale - w % mod_scale)
                lr_img = F.pad(lr_img, (0, w_pad, 0, h_pad),  "constant", 0.0) 

            # lr_img = lr_img.numpy()
            # out = predict(lr_img).squeeze() # C H W
            
            # 运行模型
            out = trt_model(lr_img).squeeze()


            # out = torch.from_numpy(out).to(device)
            if mod_scale is not None:
                _, h, w = out.shape
                out = out[:, 0:h - h_pad, 0:w - w_pad]

            out *= 255.
            out = out.round().clamp(0, 255).to(torch.uint8).transpose(0,2).transpose(0,1).cpu().numpy()
            out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
            # out = postprocess(out)

            # 将图片输出到path
            path = f'{args.output}/{(i):05d}.png'

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