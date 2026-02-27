# HoloSegment

CLI application for artery/vein segmentation from doppler holograms.

## Overview

HoloSegment processes spectral moments of doppler holograms to perform artery/vein segmentation. The processing pipeline includes:

1. **Reading .holo files**: Reads raw data with header and footer
2. **Preprocessing**: Normalization and registration of frames
3. **Binary segmentation**: Vessel mask extraction
4. **Pulse analysis**: Temporal analysis using vessel mask
5. **Semantic segmentation**: Artery/vein classification

## Installation

```bash
pip install -e .
```

Or install dependencies directly:

```bash
pip install -r requirements.txt
```

## Usage

```bash
holosegment <config.json> <input.holo> [-o output_dir] [-v]
```

### Arguments

- `config.json`: JSON configuration file with processing parameters
- `input.holo`: Input hologram file in .holo format
- `-o, --output`: Output directory for results (default: `output`)
- `-v, --verbose`: Enable verbose output

### Example

```bash
holosegment config_example.json data.holo -o results -v
```

### Configuration Parameters

#### Preprocessing
- `normalize_method`: Normalization method (`zscore`, `minmax`, `percentile`)
- `register`: Enable frame registration (boolean)
- `reference_frame`: Index of reference frame for registration

#### Binary Segmentation
- `threshold_method`: Thresholding method (`otsu`, `adaptive`, `percentile`)
- `min_vessel_size`: Minimum vessel size in pixels
- `use_temporal_variance`: Use temporal variance for segmentation (boolean)

#### Pulse Analysis
- `sampling_rate`: Sampling rate in Hz
- `frequency_range`: Physiological frequency range for analysis [min, max] in Hz

#### Semantic Segmentation
- `pulsatility_threshold`: Threshold to distinguish arteries from veins

## Output Files

The application generates the following output files:

- `vessel_mask.npy`: Binary vessel mask (numpy array)
- `pulse_results.json`: Pulse analysis metrics
- `artery_vein_mask.npy`: Semantic segmentation mask (0=background, 1=vein, 2=artery)

## License

MIT