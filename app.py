import os

apt -y update -qq
apt -y install -qq aria2
#pip install -q torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1+cu118 torchtext==0.15.1 torchdata==0.6.0 --extra-index-url https://download.pytorch.org/whl/cu118 -U
#pip install -q xformers==0.0.19 triton==2.0.0 -U
pip install -q mediapipe==0.9.1.0 addict yapf fvcore omegaconf

git clone https://github.com/comfyanonymous/ComfyUI
%cd /home/xlab-app-center/ComfyUI
pip install -q -r requirements.txt
git reset --hard
git clone https://github.com/Fannovel16/comfy_controlnet_preprocessors /home/xlab-app-center/ComfyUI/custom_nodes/comfy_controlnet_preprocessors
%cd /home/xlab-app-center/ComfyUI/custom_nodes/comfy_controlnet_preprocessors
python install.py --no_download_ckpts
%cd /home/xlab-app-center/ComfyUI

wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O /home/xlab-app-center/cloudflared-linux-amd64 && chmod 777 /home/xlab-app-center/cloudflared-linux-amd64
import atexit, requests, subprocess, time, re, os
from random import randint
from threading import Timer
from queue import Queue
def cloudflared(port, metrics_port, output_queue):
    atexit.register(lambda p: p.terminate(), subprocess.Popen(['/home/xlab-app-center/cloudflared-linux-amd64', 'tunnel', '--url', f'http://127.0.0.1:{port}', '--metrics', f'127.0.0.1:{metrics_port}'], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT))
    attempts, tunnel_url = 0, None
    while attempts < 10 and not tunnel_url:
        attempts += 1
        time.sleep(3)
        try:
            tunnel_url = re.search("(?P<url>https?:\/\/[^\s]+.trycloudflare.com)", requests.get(f'http://127.0.0.1:{metrics_port}/metrics').text).group("url")
        except:
            pass
    if not tunnel_url:
        raise Exception("Can't connect to Cloudflare Edge")
    output_queue.put(tunnel_url)
output_queue, metrics_port = Queue(), randint(8100, 9000)
thread = Timer(2, cloudflared, args=(8188, metrics_port, output_queue))
thread.start()
thread.join()
tunnel_url = output_queue.get()
os.environ['webui_url'] = tunnel_url
print(tunnel_url)

aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/control_v11e_sd15_ip2p_fp16.safetensors -d /home/xlab-app-center/ComfyUI/models/controlnet -o control_v11e_sd15_ip2p_fp16.safetensors
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/control_v11e_sd15_shuffle_fp16.safetensors -d /home/xlab-app-center/ComfyUI/models/controlnet -o control_v11e_sd15_shuffle_fp16.safetensors
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/control_v11p_sd15_canny_fp16.safetensors -d /home/xlab-app-center/ComfyUI/models/controlnet -o control_v11p_sd15_canny_fp16.safetensors
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/control_v11f1p_sd15_depth_fp16.safetensors -d /home/xlab-app-center/ComfyUI/models/controlnet -o control_v11f1p_sd15_depth_fp16.safetensors
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/control_v11p_sd15_inpaint_fp16.safetensors -d /home/xlab-app-center/ComfyUI/models/controlnet -o control_v11p_sd15_inpaint_fp16.safetensors
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/control_v11p_sd15_lineart_fp16.safetensors -d /home/xlab-app-center/ComfyUI/models/controlnet -o control_v11p_sd15_lineart_fp16.safetensors
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/control_v11p_sd15_mlsd_fp16.safetensors -d /home/xlab-app-center/ComfyUI/models/controlnet -o control_v11p_sd15_mlsd_fp16.safetensors
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/control_v11p_sd15_normalbae_fp16.safetensors -d /home/xlab-app-center/ComfyUI/models/controlnet -o control_v11p_sd15_normalbae_fp16.safetensors
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/control_v11p_sd15_openpose_fp16.safetensors -d /home/xlab-app-center/ComfyUI/models/controlnet -o control_v11p_sd15_openpose_fp16.safetensors
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/control_v11p_sd15_scribble_fp16.safetensors -d /home/xlab-app-center/ComfyUI/models/controlnet -o control_v11p_sd15_scribble_fp16.safetensors
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/control_v11p_sd15_seg_fp16.safetensors -d /home/xlab-app-center/ComfyUI/models/controlnet -o control_v11p_sd15_seg_fp16.safetensors
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/control_v11p_sd15_softedge_fp16.safetensors -d /home/xlab-app-center/ComfyUI/models/controlnet -o control_v11p_sd15_softedge_fp16.safetensors
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/control_v11p_sd15s2_lineart_anime_fp16.safetensors -d /home/xlab-app-center/ComfyUI/models/controlnet -o control_v11p_sd15s2_lineart_anime_fp16.safetensors
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/control_v11f1e_sd15_tile_fp16.safetensors -d /home/xlab-app-center/ComfyUI/models/controlnet -o control_v11f1e_sd15_tile_fp16.safetensors
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet-v1-1/raw/main/control_v11e_sd15_ip2p_fp16.yaml -d /home/xlab-app-center/ComfyUI/models/controlnet -o control_v11e_sd15_ip2p_fp16.yaml
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet-v1-1/raw/main/control_v11e_sd15_shuffle_fp16.yaml -d /home/xlab-app-center/ComfyUI/models/controlnet -o control_v11e_sd15_shuffle_fp16.yaml
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet-v1-1/raw/main/control_v11p_sd15_canny_fp16.yaml -d /home/xlab-app-center/ComfyUI/models/controlnet -o control_v11p_sd15_canny_fp16.yaml
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet-v1-1/raw/main/control_v11f1p_sd15_depth_fp16.yaml -d /home/xlab-app-center/ComfyUI/models/controlnet -o control_v11f1p_sd15_depth_fp16.yaml
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet-v1-1/raw/main/control_v11p_sd15_inpaint_fp16.yaml -d /home/xlab-app-center/ComfyUI/models/controlnet -o control_v11p_sd15_inpaint_fp16.yaml
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet-v1-1/raw/main/control_v11p_sd15_lineart_fp16.yaml -d /home/xlab-app-center/ComfyUI/models/controlnet -o control_v11p_sd15_lineart_fp16.yaml
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet-v1-1/raw/main/control_v11p_sd15_mlsd_fp16.yaml -d /home/xlab-app-center/ComfyUI/models/controlnet -o control_v11p_sd15_mlsd_fp16.yaml
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet-v1-1/raw/main/control_v11p_sd15_normalbae_fp16.yaml -d /home/xlab-app-center/ComfyUI/models/controlnet -o control_v11p_sd15_normalbae_fp16.yaml
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet-v1-1/raw/main/control_v11p_sd15_openpose_fp16.yaml -d /home/xlab-app-center/ComfyUI/models/controlnet -o control_v11p_sd15_openpose_fp16.yaml
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet-v1-1/raw/main/control_v11p_sd15_scribble_fp16.yaml -d /home/xlab-app-center/ComfyUI/models/controlnet -o control_v11p_sd15_scribble_fp16.yaml
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet-v1-1/raw/main/control_v11p_sd15_seg_fp16.yaml -d /home/xlab-app-center/ComfyUI/models/controlnet -o control_v11p_sd15_seg_fp16.yaml
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet-v1-1/raw/main/control_v11p_sd15_softedge_fp16.yaml -d /home/xlab-app-center/ComfyUI/models/controlnet -o control_v11p_sd15_softedge_fp16.yaml
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet-v1-1/raw/main/control_v11p_sd15s2_lineart_anime_fp16.yaml -d /home/xlab-app-center/ComfyUI/models/controlnet -o control_v11p_sd15s2_lineart_anime_fp16.yaml
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet-v1-1/raw/main/control_v11f1e_sd15_tile_fp16.yaml -d /home/xlab-app-center/ComfyUI/models/controlnet -o control_v11f1e_sd15_tile_fp16.yaml
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/t2iadapter_style_sd14v1.pth -d /home/xlab-app-center/ComfyUI/models/controlnet -o t2iadapter_style_sd14v1.pth
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/t2iadapter_sketch_sd14v1.pth -d /home/xlab-app-center/ComfyUI/models/controlnet -o t2iadapter_sketch_sd14v1.pth
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/t2iadapter_seg_sd14v1.pth -d /home/xlab-app-center/ComfyUI/models/controlnet -o t2iadapter_seg_sd14v1.pth
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/t2iadapter_openpose_sd14v1.pth -d /home/xlab-app-center/ComfyUI/models/controlnet -o t2iadapter_openpose_sd14v1.pth
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/t2iadapter_keypose_sd14v1.pth -d /home/xlab-app-center/ComfyUI/models/controlnet -o t2iadapter_keypose_sd14v1.pth
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/t2iadapter_depth_sd14v1.pth -d /home/xlab-app-center/ComfyUI/models/controlnet -o t2iadapter_depth_sd14v1.pth
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/t2iadapter_color_sd14v1.pth -d /home/xlab-app-center/ComfyUI/models/controlnet -o t2iadapter_color_sd14v1.pth
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/t2iadapter_canny_sd14v1.pth -d /home/xlab-app-center/ComfyUI/models/controlnet -o t2iadapter_canny_sd14v1.pth
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/t2iadapter_canny_sd15v2.pth -d /home/xlab-app-center/ComfyUI/models/controlnet -o t2iadapter_canny_sd15v2.pth
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/t2iadapter_depth_sd15v2.pth -d /home/xlab-app-center/ComfyUI/models/controlnet -o t2iadapter_depth_sd15v2.pth
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/t2iadapter_sketch_sd15v2.pth -d /home/xlab-app-center/ComfyUI/models/controlnet -o t2iadapter_sketch_sd15v2.pth
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/ControlNet-v1-1/resolve/main/t2iadapter_zoedepth_sd15v1.pth -d /home/xlab-app-center/ComfyUI/models/controlnet -o t2iadapter_zoedepth_sd15v1.pth

aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth -d /home/xlab-app-center/ComfyUI/models/upscale_models -o RealESRGAN_x2plus.pth

aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/anything-v4.0/resolve/main/anything-v4.5-pruned.ckpt -d /home/xlab-app-center/ComfyUI/models/checkpoints -o anything-v4.5-pruned.ckpt
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/anything-v4.0/resolve/main/anything-v4.0.vae.pt -d /home/xlab-app-center/ComfyUI/models/vae -o anything-v4.5-pruned.vae.pt

python main.py --dont-print-server
