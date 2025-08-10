import argparse
import time
import torch
from utils import measure_SSCD_similarity, measure_CLIP_similarity
import open_clip
from PIL import Image

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

'''
Download SSCD model in local directory 
Refer this: https://github.com/facebookresearch/sscd-copy-detection/blob/main/README.md
'''
#load models for SSCD similarity & CLIP score
sim_model = torch.jit.load("sscd_disc_large.torchscript.pt").to(device)

ref_model, _, ref_clip_preprocess = open_clip.create_model_and_transforms(
                    'ViT-H-14',
                    pretrained='laion2B_s32B_b79K',
                    device=device,
                )
ref_model.eval()
ref_tokenizer = open_clip.get_tokenizer('ViT-H-14')

''' 
BELOW code does not work !!
This is a guideline when evaluating SSCD similarity score / CLIP score when you have ground truth training images. 
Integrate the below code with "mitigate_mem.py" for automatic evaluation. 
'''

prompt = 'THIS IS DUMMY CODE'
images, prompt_embeds = pipe(prompt, num_images_per_prompt=ipp, args=args)

gt_sscd_lst = [Image.open(gt) for gt in all_gt_images[line_id]] 
gt_clip_lst = [gt_sscd_lst[0]]
gen_lst = images.images

SSCD_sim = measure_SSCD_similarity(gt_sscd_lst, gen_lst, sim_model, device)
CLIP_sim = measure_CLIP_similarity(
        gt_clip_lst + gen_lst,
        prompt,
        ref_model,
        ref_clip_preprocess,
        ref_tokenizer,
        device,
    )