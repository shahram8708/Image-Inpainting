import torch
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image, make_image_grid
import requests
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt

pipeline = AutoPipelineForInpainting.from_pretrained(
    "kandinsky-community/kandinsky-2-2-decoder-inpaint", torch_dtype=torch.float16
)
pipeline.enable_model_cpu_offload()

pipeline.enable_xformers_memory_efficient_attention()

def load_image_from_url(url):
    response = requests.get(url)
    return Image.open(BytesIO(response.content))

init_image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png"
mask_image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint_mask.png"
init_image = load_image_from_url(init_image_url)
mask_image = load_image_from_url(mask_image_url)

prompt = "a black cat with glowing eyes, cute, adorable, disney, pixar, highly detailed, 8k"
negative_prompt = "bad anatomy, deformed, ugly, disfigured"

image = pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image=init_image,
    mask_image=mask_image
).images[0]

def display_image_grid(images, rows, cols):
    grid = make_image_grid(images, rows=rows, cols=cols)
    plt.figure(figsize=(15,5))
    plt.imshow(grid)
    plt.axis("off")
    plt.show()

display_image_grid([init_image, mask_image, image], rows=1, cols=3)
