# Multi-Model Image Inpainting

This project demonstrates image inpainting using three different pre-trained models from the `diffusers` library: 
- Kandinsky Inpainting
- Stable Diffusion Inpainting (runwayml)
- Stable Diffusion XL Inpainting

Each model is used to inpaint an image with a given mask and prompt, and the results are displayed using `matplotlib`.

## Models Used:

1. **Kandinsky Inpainting**  
   Uses the model from the Kandinsky community to inpaint an image based on a prompt describing a black cat.
   
2. **Stable Diffusion Inpainting (runwayml)**  
   This model generates a fantasy castle inspired by the world of "Lord of the Rings" using inpainting.
   
3. **Stable Diffusion XL Inpainting**  
   A more advanced and highly detailed model used to generate the same elven castle in a higher quality output.

## Features
- **Inpainting**: Uses the multi pre-trained model inpainting pipeline.
- **Custom Masking**: Provides a mechanism to apply masks to an image and regenerate parts of the image based on the user's prompt.
- **Memory Efficient**: With xformers support, the pipeline is optimized for memory usage.

## How it Works

- **Initial Image**: The image you want to modify.
- **Mask**: The area of the image you want to fill with new content.
- **Prompt**: Text that describes the content you'd like to appear in the masked area.
- **Negative Prompt**: Text that describes what you want to avoid in the result (optional but useful for better results).

The pipeline uses Hugging Face's `AutoPipelineForInpainting`, a pre-trained inpainting model, to generate new content in the masked areas based on the given prompts.
