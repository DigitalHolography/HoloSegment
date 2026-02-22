# HoloSegment Workflow Guide

This guide demonstrates the complete workflow for using HoloSegment to perform artery/vein segmentation from doppler holograms.

## Quick Start

### 1. Install HoloSegment

```bash
# Clone the repository
git clone https://github.com/DigitalHolography/git
cd HoloSegment

# Install the package
pip install -e .
```

### 2. Prepare Your Data

HoloSegment requires:
- A `.holo` file containing doppler hologram frames
- A `.json` configuration file with processing parameters

### 3. Run the Processing Pipeline

```bash
holosegment config.json input.holo -o output_dir -v
```

## Detailed Workflow

### Step 1: Configure Processing Parameters

Create a configuration file (e.g., `my_config.json`):

```json
{
  "preprocessing": {
    "normalize_method": "zscore",
    "register": true,
    "reference_frame": 0
  },
  "binary_segmentation": {
    "threshold_method": "otsu",
    "min_vessel_size": 100,
    "use_temporal_variance": true
  },
  "pulse_analysis": {
    "sampling_rate": 1.0,
    "frequency_range": [0.5, 3.0]
  },
  "semantic_segmentation": {
    "pulsatility_threshold": 0.5
  }
}
```

#### Configuration Options

**Preprocessing:**
- `normalize_method`: Choose from "zscore", "minmax", or "percentile"
  - `zscore`: Z-score normalization (mean=0, std=1)
  - `minmax`: Min-max normalization to [0, 1]
  - `percentile`: Percentile-based normalization
- `register`: Enable/disable frame registration (true/false)
- `reference_frame`: Index of frame to use as reference (default: 0)

**Binary Segmentation:**
- `threshold_method`: Choose from "otsu", "adaptive", or "percentile"
  - `otsu`: Automatic threshold using Otsu's method
  - `adaptive`: Local adaptive thresholding
  - `percentile`: Threshold based on percentile value
- `min_vessel_size`: Minimum vessel size in pixels (removes small objects)
- `use_temporal_variance`: Use temporal variance for segmentation (true/false)
- `percentile`: Percentile value (only for percentile method, e.g., 75)

**Pulse Analysis:**
- `sampling_rate`: Temporal sampling rate in Hz
- `frequency_range`: [min, max] frequency range in Hz for pulse analysis

**Semantic Segmentation:**
- `pulsatility_threshold`: Threshold to distinguish arteries from veins
  - Higher pulsatility → Artery
  - Lower pulsatility → Vein

### Step 2: Run Processing

```bash
holosegment my_config.json my_data.holo -o results -v
```

Options:
- `-o, --output`: Output directory (default: "output")
- `-v, --verbose`: Enable verbose output

### Step 3: Examine Results

The application generates three output files:

1. **vessel_mask.npy**: Binary vessel mask
   ```python
   import numpy as np
   vessel_mask = np.load('results/vessel_mask.npy')
   # Shape: (height, width), dtype: uint8
   # Values: 0 (background), 1 (vessel)
   ```

2. **pulse_results.json**: Pulse analysis metrics
   ```json
   {
     "vessel_metrics": [
       {
         "vessel_id": 1,
         "area": 709.0,
         "centroid": [85.0, 85.0],
         "mean_pulsatility": 1.503,
         "std_pulsatility": 0.124,
         "mean_frequency": 0.0,
         "mean_intensity": 4.030
       }
     ]
   }
   ```

3. **artery_vein_mask.npy**: Semantic segmentation
   ```python
   import numpy as np
   semantic_mask = np.load('results/artery_vein_mask.npy')
   # Shape: (height, width), dtype: uint8
   # Values: 0 (background), 1 (vein), 2 (artery)
   ```

## Example: Complete Analysis

```bash
# 1. Create synthetic test data (for testing)
python create_test_holo.py

# 2. Run segmentation with verbose output
holosegment config_example.json /tmp/test_data.holo -o results -v

# 3. Verify the results
python verify_output.py results
```

## Processing Pipeline Details

The HoloSegment pipeline consists of five main stages:

### 1. Data Loading
- Reads `.holo` file with header and footer
- Extracts frame dimensions and metadata
- Loads all frames into memory

### 2. Preprocessing
- **Normalization**: Standardizes intensity values across frames
- **Registration**: Aligns frames using phase correlation to correct for motion

### 3. Binary Segmentation
- Computes temporal variance or mean intensity
- Applies thresholding to extract vessel regions
- Morphological operations to clean up the mask

### 4. Pulse Analysis
- Extracts temporal signals from vessel pixels
- Computes pulsatility index: (max - min) / mean
- Performs FFT to find dominant frequencies
- Calculates per-vessel metrics

### 5. Semantic Segmentation
- Labels connected vessel components
- Classifies each vessel based on pulsatility:
  - High pulsatility → Artery
  - Low pulsatility → Vein

## Tuning Parameters

### For Better Vessel Detection:
- Adjust `threshold_method` and `min_vessel_size` in binary segmentation
- Try different normalization methods
- Enable/disable frame registration based on data quality

### For Better Artery/Vein Classification:
- Adjust `pulsatility_threshold` based on your data
- Modify `frequency_range` to match physiological expectations
- Ensure `sampling_rate` matches your acquisition rate

## Troubleshooting

**Problem**: Few or no vessels detected
- Solution: Lower `min_vessel_size` or try different `threshold_method`

**Problem**: Too many false positives
- Solution: Increase `min_vessel_size` or adjust threshold parameters

**Problem**: Poor artery/vein classification
- Solution: Adjust `pulsatility_threshold` or check `sampling_rate`

**Problem**: Registration fails or introduces artifacts
- Solution: Set `register: false` in configuration

## Advanced Usage

### Custom Processing

You can use individual modules programmatically:

```python
from reader import HoloReader
from preprocessing import preprocess_frames
from segmentation import binary_segmentation

# Load data
reader = HoloReader('data.holo')
frames = reader.read_frames()

# Preprocess
config = {'normalize_method': 'zscore', 'register': True}
preprocessed = preprocess_frames(frames, config)

# Segment
vessel_mask = binary_segmentation(preprocessed, {'threshold_method': 'otsu'})
```

### Batch Processing

```bash
for file in data/*.holo; do
    base=$(basename "$file" .holo)
    holosegment config.json "$file" -o "results/$base"
done
```

## Citation

If you use HoloSegment in your research, please cite:

```
[Citation information to be added]
```

## Support

For issues, questions, or contributions, please visit:
https://github.com/DigitalHolography/HoloSegment
