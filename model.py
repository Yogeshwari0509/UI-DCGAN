
import streamlit as st
import numpy as np
import torch
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
from PIL import Image
import tensorflow as tf
import h5py
import json

SEED_SIZE = 100
EMBEDDING_SIZE = 300

def clean_h5_model(file_path):
    with h5py.File(file_path, 'r+') as h5_file:
        model_config = h5_file.attrs['model_config']
        if isinstance(model_config, bytes):
            model_config = model_config.decode('utf-8')
        model_config = json.loads(model_config)

        for layer in model_config['config']['layers']:
            if layer['class_name'] == 'Conv2DTranspose' and 'groups' in layer['config']:
                del layer['config']['groups']
        h5_file.attrs['model_config'] = json.dumps(model_config).encode('utf-8')

@st.cache_resource
def load_gan_model(model_path):
    clean_h5_model(model_path)
    return tf.keras.models.load_model(model_path)

def generate_image(gan_model, text_input):
    noise = tf.random.normal([1, SEED_SIZE])
    words = text_input.lower().split()
    test_embeddings = np.zeros((1, EMBEDDING_SIZE), dtype=np.float32)
    count = 0

    for word in words:
        if word in glove_embeddings:
            test_embeddings[0] += glove_embeddings[word]
            count += 1

    if count > 0:
        test_embeddings[0] /= count

    generated_image = gan_model.predict([noise, test_embeddings], verbose=0)
    generated_image = 0.5 * generated_image + 0.5
    image = Image.fromarray((generated_image[0] * 255).astype(np.uint8)) 
    
    # Assign the image to low_quality_image before resizing it
    low_quality_image = image.resize((32, 32), Image.NEAREST)
    
    return low_quality_image

@st.cache_resource
def load_pipeline():
    model_id = "stabilityai/stable-diffusion-2"
    scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
    pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, revision="fp16", torch_dtype=torch.float16)
    return pipe.to("cuda")

def modify_image_with_stable_diffusion(image, prompt):
    pipe = load_pipeline()
    image = image.convert("RGB").resize((32, 32))

    refined_image = pipe(prompt=prompt, init_image=image, strength=0.1, num_inference_steps=10).images[0]
    low_quality_refined_image = refined_image.resize((32, 32), Image.NEAREST)
    
    return low_quality_refined_image

st.title('Text to Image Generator with GAN')

model_path = '/content/text_to_image_generator_cub_character.h5'
gan_model = load_gan_model(model_path)
st.success("GAN model loaded successfully!")

glove_embeddings = np.load('/content/embedding_data.npy', allow_pickle=True)
glove_embeddings = {item[0]: item[1] for item in glove_embeddings} if isinstance(glove_embeddings, np.ndarray) else glove_embeddings.item()
st.success("GloVe embeddings loaded successfully!")

text_input = st.text_input("What image would you like to create?")

if text_input and gan_model:
    generated_image = generate_image(gan_model, text_input)
    refined_image = modify_image_with_stable_diffusion(generated_image, text_input)
    st.image(refined_image, caption='Generated Image from GAN', use_column_width=True)
