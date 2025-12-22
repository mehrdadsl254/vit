# Qwen 2.5 VL Patch Similarity Experiment

Analyze cosine similarity between image patches at different transformer layers in Qwen 2.5 VL 7B-Instruct.

## Two Experiments

1. **Encoder Experiment** - Analyzes ViT (Vision Transformer) encoder layers
2. **Decoder Experiment** - Analyzes LLM decoder layers where vision tokens are processed

## Quick Start

### On GPU Server (Feature Extraction)

```bash
# Install dependencies
pip install -r requirements.txt

# Run encoder experiment
python run_encoder_experiment.py --color green --output outputs/encoder/

# Run decoder experiment
python run_decoder_experiment.py --color green --output outputs/decoder/
```

### Save Features Only (Run on server, visualize locally)

```bash
# On server: extract features only
python run_encoder_experiment.py --color green --features-only --output outputs/encoder/
python run_decoder_experiment.py --color green --features-only --output outputs/decoder/

# Copy features to local machine, then:
python visualize_offline.py --features outputs/encoder/features/green_surface_encoder_features.npz \
                            --image outputs/encoder/green_surface.png \
                            --component encoder \
                            --output outputs/local/
```

## Configuration

Edit `config.py` to modify:
- **Layers**: Automatically generated with pattern (1-5 by 1, then by 2, then by 3)
- **Patch positions**: 6 locations (Center, N, S, E, W, NW)
- **Image size**: Default 448x448

## Output

The visualization shows:
- **Rows** = Different transformer layers
- **Columns** = Different selected patches
- **Each cell** = Image with patch grid overlay
  - Red box = Selected patch
  - Color intensity = Cosine similarity (green=high, red=low)
  - Numbers = Similarity values

## File Structure

```
Mllp/
├── config.py                  # Configuration
├── generate_image.py          # Create test images
├── feature_extractor.py       # Model hooks for feature extraction
├── similarity.py              # Cosine similarity computation
├── visualize.py               # Multi-panel visualization
├── run_encoder_experiment.py  # Encoder experiment script
├── run_decoder_experiment.py  # Decoder experiment script
├── visualize_offline.py       # Visualize from saved features
├── requirements.txt           # Dependencies
└── outputs/                   # Generated outputs
    ├── encoder/
    │   ├── features/
    │   └── figures/
    └── decoder/
        ├── features/
        └── figures/
```
