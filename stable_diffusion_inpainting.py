import torch
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image, make_image_grid
import requests
from PIL import Image
import io
import matplotlib.pyplot as plt

def download_image(url):
    response = requests.get(url)
    if response.status_code == 200: 
        return Image.open(io.BytesIO(response.content))
    else:
        raise Exception(f"Failed to download image from {url}")

init_image = download_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png")
mask_image = download_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint_mask.png")

pipeline = AutoPipelineForInpainting.from_pretrained(
    "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16, variant="fp16"
)

pipeline.enable_model_cpu_offload()

generator = torch.Generator(device="cuda").manual_seed(92)
prompt = "concept art digital painting of an elven castle, inspired by lord of the rings, highly detailed, 8k"

image = pipeline(prompt=prompt, image=init_image, mask_image=mask_image, generator=generator).images[0]

def display_images(images, titles=None):
    plt.figure(figsize=(15,5))
    for i, img in enumerate(images):
        plt.subplot(1, len(images), i + 1)
        plt.imshow(img)
        plt.axis("off")
        if titles:
            plt.title(titles[i])

display_images([init_image, mask_image, image], titles=["Original Image", "Mask Image", "Inpainted Image"])
plt.show()
