import os
import cv2
import torch
import argparse
import numpy as np
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler,T2IAdapter,StableDiffusionAdapterPipeline
from ControlNet.annotator.util import resize_image, HWC3
from ControlNet.annotator.canny import CannyDetector
from annotator1.content import ContentDetector
import einops
from pytorch_lightning import seed_everything

# ======== Uni_ControlNet 初始化 ========
from Uni_ControlNet.models.util import create_model, load_state_dict
from Uni_ControlNet.models.ddim_hacked import DDIMSampler
import Uni_ControlNet.utils.config as config
import subprocess


# 初始化Canny边缘检测器
apply_canny = CannyDetector()
apply_content = ContentDetector()



def load_unicontrolnet(config_path, ckpt_path, device='cuda'):
    model = create_model(config_path).cpu()
    model.load_state_dict(load_state_dict(ckpt_path, location=device), strict=False)
    model = model.to(device)
    sampler = DDIMSampler(model)
    return model, sampler


# 加载 pipeline
def load_pipeline(method='controlnet', torch_dtype=torch.float16, device='cuda'):

    controlnet_path="E:\phd//3\code\ppi_control\lllyasvielsd-controlnet-scribble"
    controlnet_plusplus_path='E:\phd//3\code\ppi_control\controlnet_plusplus'
    sd_path="E:\phd//3\code\ppi_control\stable-diffusion-v1-5stable-diffusion-v1-5"
    t2i_adapter_path = "E:\phd//3\code\ppi_control\TencentARC_t2iadapter_canny_sd15v2"
    uni_config_path = "E:\phd//3\code\ppi_control//Uni_ControlNet\configs//uni_v15.yaml"
    uni_ckpt_path = "E:\phd//3\code\ppi_control//Uni_ControlNet\ckpt//uni.ckpt"

    if method == 't2i_adapter':
        adapter = T2IAdapter.from_pretrained(t2i_adapter_path, torch_dtype=torch_dtype)
        pipe = StableDiffusionAdapterPipeline.from_pretrained(
            sd_path,
            adapter=adapter,
            torch_dtype=torch_dtype
        ).to(device)
    elif method == 'controlnet':
        #####修改路径即可controlnet_plusplus_path#########
        controlnet = ControlNetModel.from_pretrained(controlnet_plusplus_path, torch_dtype=torch_dtype)
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            sd_path,
            controlnet=controlnet,
            torch_dtype=torch_dtype,
            safety_checker=None
        ).to(device)
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)


    elif method == 'unicontrolnet':
        model, sampler = load_unicontrolnet(
            config_path = uni_config_path,
            ckpt_path = uni_ckpt_path
        )
        return ('unicontrolnet', model, sampler)

    else:
        raise ValueError(f"Unsupported method: {method}")

    return pipe




def controlar(
    input_image_path,
    full_prompt,
    output_path,
    output_name,
    seed,
    conda_env_name="ControlAR",  # 目标虚拟环境名
    target_script="E:\phd//3\code\ControlAR-main//autoregressive\sample\sample_t2i.py"  # 被调用的完整脚本路径
):

    # env = os.environ.copy()
    # env['PYTHONPATH'] = "E:\phd//3\code\ControlAR-main"

    cmd = [
        "conda", "run", "-n", conda_env_name, "python", target_script,
        "--condition-path", input_image_path,
        "--prompt", full_prompt,
        "--output_path", output_path,
        "--output_name", output_name,
        "--seed", str(seed),
    ]


    print("[controlar] Executing external command:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)



# 定义图像处理函数
def run_controlnet(
    input_image_path,
    prompt,
    a_prompt=" American comic art style, best quality, masterpiece,Simple background,",
        #####Qi Baishi style, Van Gogh style,，super - deformed anime style, Anime character painted in watercolor style，Chibi - style character, super deformed anime style,
    n_prompt="longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality",
    method='controlnet',
    num_samples=1,
    image_resolution=512,
    ddim_steps=50,
    guess_mode=False,
    strength=1.0,
    scale=10,
    seed=4,
    eta=0.0,
    low_threshold=100,
    high_threshold=200,
    output_path='E:/phd/3/code/ppi_control/img/test/',
    output_name='other',
):

    os.makedirs(output_path, exist_ok=True)
    # 拼接 prompt
    full_prompt = f"{a_prompt}, {prompt}"

    if method == "controlar":
        return controlar(
            input_image_path=input_image_path,
            full_prompt=full_prompt,
            output_path=output_path,
            output_name=output_name,
            seed=seed,
            conda_env_name="ControlAR",  # 👈 修改为你目标环境名
            target_script="E:\phd//3\code\ControlAR-main//autoregressive\sample\sample_t2i.py"  # 👈 修改为目标脚本路径
        )



    # 加载 pipeline
    pipe = load_pipeline(method=method)

    # 设置 seed
    if seed != -1:
        generator = torch.manual_seed(seed)
    else:
        generator = torch.Generator(device='cuda')


    if method == 'unicontrolnet':
        _, model, sampler = pipe
        input_image = cv2.imread(input_image_path)
        H, W, C = input_image.shape
        canny_image = cv2.resize(input_image, (W, H))
        canny_detected_map = HWC3(apply_canny(HWC3(canny_image), low_threshold, high_threshold))


        # content embedding 这里需要你自己的 apply_content 函数
        content_image = False
        if content_image == True:
            content_emb = apply_content(input_image)  # shape: (768,)
        else:
            content_emb = np.zeros((768), dtype=np.float32)



        detected_map = np.zeros((H, W, C), dtype=np.uint8)
        detected_maps_list = [canny_detected_map,
                              detected_map,
                              detected_map,
                              detected_map,
                              detected_map,
                              detected_map,
                              detected_map
                              ]
        detected_maps = np.concatenate(detected_maps_list, axis=2)


        local_control = torch.from_numpy(detected_maps.copy()).float().cuda() / 255.0
        local_control = torch.stack([local_control for _ in range(num_samples)], dim=0)
        local_control = einops.rearrange(local_control, 'b h w c -> b c h w').clone()
        global_control = torch.from_numpy(content_emb.copy()).float().cuda().clone()
        global_control = torch.stack([global_control for _ in range(num_samples)], dim=0)

        uc_local_control = local_control
        uc_global_control = torch.zeros_like(global_control)

        # 构建条件 & 负条件
        cond = {
            "local_control": [local_control],
            "c_crossattn": [model.get_learned_conditioning([full_prompt] * num_samples)],
            "global_control": [global_control]
        }
        un_cond = {
            "local_control": [uc_local_control],
            "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)],
            "global_control": [uc_global_control]
        }

        # 设置控制强度（13层）
        model.control_scales = [strength] * 13

        shape = (4, H // 8, W // 8)
        samples, _ = sampler.sample(
            S=ddim_steps,
            batch_size=num_samples,
            shape=shape,
            conditioning=cond,
            unconditional_guidance_scale=scale,
            unconditional_conditioning=un_cond,
            eta=eta,
            x_T=None,
            global_strength=strength,
        )

        x_samples = model.decode_first_stage(samples).cpu()
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).numpy().clip(0, 255).astype(
            np.uint8)
        results = [Image.fromarray(img) for img in x_samples]


    else:

        # 加载图像并应用 Canny
        input_image = cv2.imread(input_image_path)
        detected_map = apply_canny(input_image, low_threshold, high_threshold)
        control_image = Image.fromarray(detected_map)

        # control_image.save(os.path.join(output_path, f"canny_{output_name}.png"))

        # input_image = cv2.imread(input_image_path)
        # H, W, C = input_image.shape
        # canny_image = cv2.resize(input_image, (W, H))
        # control_image = HWC3(apply_canny(HWC3(canny_image), low_threshold, high_threshold))

        # 通用调用接口（根据方法动态配置）
        pipe_args = {
            "prompt": [full_prompt] * num_samples,
            "negative_prompt": [n_prompt] * num_samples,
            "image": control_image,
            "num_inference_steps": ddim_steps,
            "guidance_scale": scale,
            "generator": generator,
            }

        if method == 'controlnet':
                pipe_args["controlnet_conditioning_scale"] = strength
        elif method == 't2i_adapter':
                pipe_args["adapter_conditioning_scale"] = strength

        outputs = pipe(**pipe_args)
        results = outputs["images"]


    for i, image in enumerate(results):
        image.save(os.path.join(output_path, f"controlnet_{output_name}.png"))

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Control Stable Diffusion with Canny Edge Maps')
    parser.add_argument('--input_image_path', type=str , default='E:\phd//3\code\ppi_control\inputs\image_299.png', help='Path to the input image')
    parser.add_argument('--prompt', type=str , default='A light sofa with two cushions, an wall painting, and a small round side table with a vase of flowers.', help='Prompt for the model')
    parser.add_argument('--a_prompt', type=str, default='Anime key visual style,best quality, masterpiece,Simple background,', help='Additional prompt')
    parser.add_argument('--n_prompt', type=str,default='fewer digits, cropped,low quality',help='Negative prompt')
    parser.add_argument('--method', type=str, default='controlnet',choices=['controlnet', 'ti_adapter','unicontrolnet','controlar'],help='Choose the control t2i method')
    parser.add_argument('--output_path', type=str , default='E:\phd//3\code\ppi_control\data/control_privacy', help='Output path prefix')
    parser.add_argument('--output_name', type=str, default='image_299_Animekeyvisual',help='Output image name')

    parser.add_argument('--num_samples', type=int, default=1, help='Number of samples to generate')
    parser.add_argument('--image_resolution', type=int, default=512, help='Image resolution')
    parser.add_argument('--ddim_steps', type=int, default=50, help='Number of DDIM steps')
    parser.add_argument('--guess_mode', action='store_true', help='Enable guess mode')
    parser.add_argument('--strength', type=float, default=1.0, help='Control strength')
    parser.add_argument('--scale', type=float, default=7.5, help='Guidance scale')
    parser.add_argument('--seed', type=int, default=4, help='Random seed')
    parser.add_argument('--eta', type=float, default=0.0, help='DDIM eta')
    parser.add_argument('--low_threshold', type=int, default=100, help='Canny low threshold')
    parser.add_argument('--high_threshold', type=int, default=200, help='Canny high threshold')


    args = parser.parse_args()

    run_controlnet(
        input_image_path=args.input_image_path,
        prompt=args.prompt,
        a_prompt=args.a_prompt,
        n_prompt=args.n_prompt,
        method=args.method,
        num_samples=args.num_samples,
        image_resolution=args.image_resolution,
        ddim_steps=args.ddim_steps,
        guess_mode=args.guess_mode,
        strength=args.strength,
        scale=args.scale,
        seed=args.seed,
        eta=args.eta,
        low_threshold=args.low_threshold,
        high_threshold=args.high_threshold,
        output_path=args.output_path,
        output_name=args.output_name,
    )