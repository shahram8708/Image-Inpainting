## Kandinsky Image Inpainting

### Overview
This repository contains an inpainting pipeline using the [Kandinsky 2.2 Decoder](https://huggingface.co/kandinsky-community/kandinsky-2-2-decoder-inpaint) from Hugging Face. Inpainting allows you to modify or fill missing parts of an image based on a given prompt.

The pipeline leverages `torch` for deep learning, `diffusers` from Hugging Face for handling inpainting tasks, and `xformers` for efficient memory usage.

### Features
- **Inpainting with Kandinsky 2.2**: Uses the pre-trained `Kandinsky` inpainting pipeline.
- **Custom Masking**: Provides a mechanism to apply masks to an image and regenerate parts of the image based on the user's prompt.
- **Memory Efficient**: With xformers support, the pipeline is optimized for memory usage.

### How it Works

- **Initial Image**: The image you want to modify.
- **Mask**: The area of the image you want to fill with new content.
- **Prompt**: Text that describes the content you'd like to appear in the masked area.
- **Negative Prompt**: Text that describes what you want to avoid in the result (optional but useful for better results).

The pipeline uses Hugging Face's `AutoPipelineForInpainting`, a pre-trained inpainting model, to generate new content in the masked areas based on the given prompts.
