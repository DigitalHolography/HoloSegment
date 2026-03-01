# HoloSegment

HoloSegment is a segmentation pipeline for retinal artery and vein extraction from Doppler holograms produced by Holodoppler systems.

The application provides:
- A modular segmentation pipeline
- A CLI interface
- A Streamlit-based GUI
- A model registry for reproducible inference

## Scientific Background

This work is based on the methodology described in:

> Author et al., *Title of the paper*, Journal, Year.
> [Link to article]

## Application Overview

HoloSegment performs:

1. Loading and preprocessing of Doppler hologram data
2. Extraction of full-field M0 images
3. Artery and vein segmentation
4. Post-processing and mask refinement
5. Optional export of results

## Installation

### Clone repository

git clone https://github.com/your_org/holosegment.git
cd holosegment

### Create environment

python -m venv .venv
source .venv/bin/activate

### Install in editable mode

pip install -e .

## CLI Usage

holosegment --help

Example:

holosegment holodoppler_folder_path -v

## GUI Usage

streamlit run app.py

## Documentation

- Architecture & pipeline design → see WORKFLOW.md
- Contribution guidelines → see CONTRIBUTING.md