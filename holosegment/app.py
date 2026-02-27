import streamlit as st
import numpy as np
import cv2
from pathlib import Path

from holosegment.pipeline.pipeline import Pipeline

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

folder = st.text_input("Folder path")

if st.button("Load Folder"):

    if not folder:
        st.warning("Please enter a folder path.")
    else:
        st.session_state.input_folder = folder

        pipeline = Pipeline(config="/d/test/250912/250912_DEA_L_HD_1/eyeflow/json/input_EF_params_0.json")
        st.session_state.pipeline = pipeline

        # Run only required steps
        pipeline.run(folder, targets=[
            "load_moments",
            "preprocess"
        ])

        image = pipeline.ctx.get("M0_ff_image")

        st.session_state.image = image
        st.success("Folder loaded and preprocessed.")


def overlay_masks(image, artery_mask=None, vein_mask=None):
    img = image.copy()

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
        pipeline.run_all(
            st.session_state.input_folder,
            targets=None  # full pipeline
        )

        st.session_state.artery_mask = pipeline.ctx.get("artery_mask")
        st.session_state.vein_mask = pipeline.ctx.get("vein_mask")

        st.success("Pipeline completed.")