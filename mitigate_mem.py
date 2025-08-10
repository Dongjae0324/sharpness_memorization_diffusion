import argparse
from tqdm import tqdm

import torch
import os
from local_model.pipe import LocalStableDiffusionPipeline
from diffusers import DDIMScheduler, UNet2DConditionModel


def main(args):
    torch.set_default_dtype(torch.bfloat16)
    used_dtype = torch.bfloat16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if args.sd_ver == 1:
        model_id = 'CompVis/stable-diffusion-v1-4'
        unet = UNet2DConditionModel.from_pretrained(
            model_id, subfolder='unet', torch_dtype=used_dtype
        )
        pipe = LocalStableDiffusionPipeline.from_pretrained(
            model_id,
            unet=unet,
            torch_dtype=used_dtype,
            safety_checker=None,
        )
    else:
        model_id = 'stabilityai/stable-diffusion-2'
        pipe = LocalStableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=used_dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
        
    args.exp_type = 'miti' #mitigation task
    gen_img_path = f'./miti_outputs/sd{args.sd_ver}/{args.prompt_type}'
    os.makedirs(gen_img_path, exist_ok=True)

    with open(args.data_path, 'r') as file:
        for line_id, line in enumerate(file):
            prompt = line.strip() 
            
            print(prompt)
            
            image_name = prompt.replace('/', '').replace('\\', '')  # Remove any slashes
            while image_name.startswith('"') or image_name.startswith("'"):
                image_name = image_name.strip('"').strip('"').strip("'").strip("'")  # Remove any leading/trailing quotes
            
            images = pipe(
                prompt,
                num_images_per_prompt=args.gen_num,
                args=args
            )
            gen_lst = images.images    
            for k in range(args.gen_num):
                gen_lst[k].save(f"{gen_img_path}/{image_name}_{k}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="diffusion memorization")
    parser.add_argument("--sd_ver", default=1, type=int)
    parser.add_argument("--gen_num", default=4, type=int)
    parser.add_argument("--gen_seed", default=42, type=int)
    parser.add_argument("--prompt_type", default='mem', type=str)
    parser.add_argument("--data_path", default='prompts/sample_mitigation.txt', type=str)
    
    ## Hyperparameters (check Appendix D)
    parser.add_argument("--miti_thres", default=8.2, type=float, help='l_thres (refer to Algorithm 2)')
    parser.add_argument("--miti_lr", default=0.05, type=float, help='learning rate for latent optimization')
    parser.add_argument("--miti_budget", default=8, type=int, help='batch size for simultaneous latent optimization')
    parser.add_argument("--miti_max_steps", default=10, type=int, help='max steps for latent optimization (may hurt gaussianity)')
    

    args = parser.parse_args()
    main(args)
