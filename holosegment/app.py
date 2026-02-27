import streamlit as st
import numpy as np
import cv2
from pathlib import Path
import tkinter as tk
from tkinter import filedialog
import json

from holosegment.pipeline.pipeline import Pipeline
from holosegment.models.registry import ModelRegistryConfig


def load_config():
    """Load configuration from JSON file"""
    config_path = select_file()
    if config_path is None or not Path(config_path).exists():
        st.warning("Please select a valid configuration file.")
        return None
    if config_path.suffix != ".json":
        st.warning("Please select a JSON configuration file.")
        return None 
    with open(config_path, 'r') as f:
        return json.load(f)
    
def select_file():
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    file_path = filedialog.askopenfilename(master=root)
    root.destroy()
    return Path(file_path)

def select_folder():
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    folder_path = filedialog.askdirectory(master=root)
    root.destroy()
    return Path(folder_path) 

def init_session():
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = None

    if "input_folder" not in st.session_state:
        st.session_state.input_folder = None

    if "image" not in st.session_state:
        st.session_state.image = None

    if "artery_mask" not in st.session_state:
        st.session_state.artery_mask = None

    if "vein_mask" not in st.session_state:
        st.session_state.vein_mask = None


init_session()

if st.button("Load config"):
    st.session_state.config = load_config()
    st.success("Config loaded.")

# 1. BROWSER BUTTON
if st.button("Browse Folder"):
    selected_path = select_folder()
        
    if not selected_path:
        st.warning("Please select a folder path.")
    else:
        st.session_state.input_folder = selected_path

        registry = ModelRegistryConfig(Path("models.yaml"))
        pipeline = Pipeline(config=st.session_state.config, model_registry=registry, output_dir=Path("output"), debug=True)
        st.session_state.pipeline = pipeline

        # Run only required steps
        pipeline.run(selected_path, targets=[
            "load_moments",
            "preprocess"
        ])

        image = pipeline.ctx.get("M0_ff_image")

        st.session_state.image = image
        st.success("Folder loaded and preprocessed.")


def overlay_masks(image, artery_mask=None, vein_mask=None):
    img = image.copy()

    print(f"Image shape: {img.shape}")
    print(image)

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    if artery_mask is not None:
        img[artery_mask > 0] = [255, 0, 0]  # red

    if vein_mask is not None:
        img[vein_mask > 0] = [0, 0, 255]  # blue

    return img

if st.session_state.image is not None:

    display_img = overlay_masks(
        st.session_state.image,
        st.session_state.artery_mask,
        st.session_state.vein_mask,
    )

    st.image(display_img, caption="M0_ff_image with overlays")


if st.button("Run Full Pipeline"):

    pipeline = st.session_state.pipeline

    if pipeline is None:
        st.warning("Load a folder first.")
    else:
        pipeline.run(
            st.session_state.input_folder,
            targets=None  # full pipeline
        )

        st.session_state.artery_mask = pipeline.ctx.get("artery_mask")
        st.session_state.vein_mask = pipeline.ctx.get("vein_mask")

        st.success("Pipeline completed.")