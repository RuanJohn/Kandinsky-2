import os
from io import BytesIO

import gradio as gr
import requests
import torch
from PIL import Image

from kandinsky2 import get_kandinsky2

# Set the cache directory (set by the Dockerfile)
CACHE_DIR = os.environ.get("KANDINSKY_CACHE_DIR", "/local/kandinsky2")

# Get the device from environment variable, default to 'cuda'
DEVICE = os.environ.get("DEVICE", "cuda")

# Load the Kandinsky2 model
model = get_kandinsky2(
    device=DEVICE,
    task_type="text2img",
    cache_dir=CACHE_DIR,
    model_version="2.1",
    use_flash_attention=True,
)


# Function to load an image from a URL
def load_image_from_url(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content)).convert("RGB")
    return img


# Function to mix images based on slider value
def mix_images(image1_url, image1_upload, image2_url, image2_upload, slider_value):
    # For Image 1
    if image1_upload is not None and image1_url.strip() != "":
        raise gr.Error(
            "Please provide only one input for Image 1: either an uploaded image or a URL."
        )
    elif image1_upload is not None:
        image1 = image1_upload
    elif image1_url.strip() != "":
        image1 = load_image_from_url(image1_url)
    else:
        raise gr.Error(
            "Please provide an input for Image 1: either an uploaded image or a URL."
        )

    # For Image 2
    if image2_upload is not None and image2_url.strip() != "":
        raise gr.Error(
            "Please provide only one input for Image 2: either an uploaded image or a URL."
        )
    elif image2_upload is not None:
        image2 = image2_upload
    elif image2_url.strip() != "":
        image2 = load_image_from_url(image2_url)
    else:
        raise gr.Error(
            "Please provide an input for Image 2: either an uploaded image or a URL."
        )

    # Deduce strengths
    strength_image1 = 1 - slider_value
    strength_image2 = slider_value

    # Run the model
    with torch.no_grad():
        image_mixed = model.mix_images(
            [image1, image2],
            [strength_image1, strength_image2],
            num_steps=100,
            batch_size=1,
            guidance_scale=4,
            h=512,  # Adjust resolution if needed
            w=512,
            sampler="p_sampler",
            prior_cf_scale=4,
            prior_steps="5",
        )[0]
    return image_mixed


# Define the Gradio interface
interface = gr.Interface(
    fn=mix_images,
    inputs=[
        gr.Textbox(label="Image 1 URL", lines=1, placeholder="Enter URL of Image 1"),
        gr.Image(type="pil", label="Upload Image 1"),
        gr.Textbox(label="Image 2 URL", lines=1, placeholder="Enter URL of Image 2"),
        gr.Image(type="pil", label="Upload Image 2"),
        gr.Slider(
            label="Image 2 Strength",
            minimum=0.0,
            maximum=1.0,
            value=0.5,
            step=0.01,
            info="Adjust the slider to set the strength of Image 2 in the mix.",
        ),
    ],
    outputs=gr.Image(label="Mixed Image"),
    title="Image Mixer",
    description="Adjust the slider to mix two images. For each image, provide either an uploaded image or a URL.",
)

# Launch the interface
if __name__ == "__main__":
    # Launch the interface
    interface.launch(share=True)
