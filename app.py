import torch
import streamlit as st
import os
from diffusers import StableDiffusionXLPipeline
from PIL import Image
import gc
import base64
from pathlib import Path

st.set_page_config(page_title="AI Image Generator", layout="wide")

def get_base64_of_bin_file(file_path):
    with open(file_path, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background():
    bin_str = get_base64_of_bin_file('bg_image.jpg')
    page_bg_img = f'''
    <style>
        .stApp {{
            background-image: linear-gradient(
                rgba(0, 26, 51, 0.95),
                rgba(0, 0, 0, 0.95)
            ), url("data:image/jpg;base64,{bin_str}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        [data-testid="stHeader"] {{
            background: transparent;
        }}
        [data-testid="stSidebar"], .main {{
            background: transparent !important;
        }}
        .stTextArea textarea, .stTextInput input {{
            background-color: rgba(0, 34, 68, 0.7) !important;
            color: white !important;
            border-radius: 6px;
            border: none !important;
            box-shadow: 0 0 10px rgba(0, 170, 255, 0.2);
        }}
        div[data-baseweb="select"] {{
            background-color: rgba(0, 34, 68, 0.7) !important;
            border-radius: 6px;
            border: none !important;
            box-shadow: 0 0 10px rgba(0, 170, 255, 0.2);
        }}
        div[data-baseweb="select"] * {{
            background-color: rgba(0, 34, 68, 0.7) !important;
            color: white !important;
            border: none !important;
        }}
        div[data-baseweb="popover"] {{
            background-color: rgba(0, 26, 51, 0.9) !important;
            border: none !important;
        }}
        div[data-baseweb="popover"] * {{
            background-color: rgba(0, 26, 51, 0.9) !important;
            color: white !important;
            border: none !important;
        }}
        .stButton>button {{
            background: rgba(0, 123, 255, 0.8);
            color: white;
            font-size: 16px;
            padding: 10px 15px;
            border-radius: 6px;
            transition: 0.3s ease-in-out;
            border: none;
            box-shadow: 0 0 10px rgba(0, 123, 255, 0.2);
        }}
        .stButton>button:hover {{
            background: rgba(0, 64, 128, 0.9);
            box-shadow: 0 0 15px rgba(0, 123, 255, 0.4);
        }}
        .stSlider {{
            color: #00aaff !important;
        }}
        .stFileUploader {{
            border: none !important;
            padding: 15px;
            border-radius: 6px;
            background: rgba(0, 34, 68, 0.7);
            box-shadow: 0 0 10px rgba(0, 170, 255, 0.2);
        }}
        .uploadedImage img {{
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0, 170, 255, 0.3);
            margin-top: 15px;
        }}
        .download-section {{
            background: rgba(0, 34, 68, 0.7);
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 0 15px rgba(0, 170, 255, 0.2);
            margin-top: 20px;
        }}
        h1, h2, h3 {{
            color: #00aaff;
            text-align: center;
        }}
        .expander-content {{
            background: rgba(0, 34, 68, 0.7) !important;
            border-radius: 6px;
            padding: 10px;
        }}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

set_background()

@st.cache_resource
def load_pipeline():
    model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    )
    if torch.cuda.is_available():
        pipe.to("cuda")
        pipe.enable_model_cpu_offload()
        pipe.enable_vae_slicing()
        torch.backends.cudnn.benchmark = True
    return pipe

pipe = load_pipeline()

def clear_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

st.markdown("<h1>AI Image Generator</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px; color: white;'>Generate high-quality AI images using Stable Diffusion XL</p>", unsafe_allow_html=True)

prompt = st.text_area("Enter Image Description:", placeholder="Describe the image you want to generate")

styles = {
    "Standard": "",
    "Photorealistic": "professional photograph, 8k UHD, detailed, sharp focus",
    "Artistic": "artistic, creative, detailed artwork, professional",
    "Digital Art": "digital art, highly detailed, professional illustration",
    "Cinematic": "cinematic, movie scene, dramatic lighting, highly detailed"
}
style_choice = st.selectbox("Choose a Style", list(styles.keys()))

negative_prompt = st.text_input("Elements to Exclude", 
    value="blurry, low quality, distorted, deformed, ugly, poor details")

with st.expander("Advanced Configuration"):
    st.markdown('<div class="expander-content">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        num_inference_steps = st.slider("Quality Steps", 20, 100, 30)
        guidance_scale = st.slider("Prompt Adherence", 5.0, 15.0, 7.5)
    with col2:
        width = st.select_slider("Width", options=[512, 768, 1024], value=1024)
        height = st.select_slider("Height", options=[512, 768, 1024], value=1024)
    st.markdown('</div>', unsafe_allow_html=True)

uploaded_image = st.file_uploader("Upload a Reference Image (Optional)", type=["png", "jpg", "jpeg"])
init_image = Image.open(uploaded_image).convert("RGB") if uploaded_image else None

if st.button("Generate Image"):
    try:
        with st.spinner("Generating image... Please wait..."):
            styled_prompt = f"{prompt}, {styles[style_choice]}" if styles[style_choice] else prompt
            
            image = pipe(
                prompt=styled_prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                image=init_image if init_image else None
            ).images[0]

            clear_memory()
            
            image_path = "generated_image.png"
            image.save(image_path)
            
            st.markdown("<div class='uploadedImage'>", unsafe_allow_html=True)
            st.image(image_path, use_column_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("<div class='download-section'>", unsafe_allow_html=True)
            st.markdown("<h3>Download Options</h3>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                with open(image_path, "rb") as file:
                    st.download_button(
                        "Download PNG",
                        file,
                        file_name="generated_image.png",
                        mime="image/png"
                    )
            with col2:
                jpeg_path = "generated_image.jpg"
                image.convert('RGB').save(jpeg_path, 'JPEG', quality=95)
                with open(jpeg_path, "rb") as file:
                    st.download_button(
                        "Download JPEG",
                        file,
                        file_name="generated_image.jpg",
                        mime="image/jpeg"
                    )
            st.markdown("</div>", unsafe_allow_html=True)
                
    except Exception as e:
        st.error(f"Generation failed: {str(e)}")
    finally:
        clear_memory()