# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Core Commands

### Training and Evaluation
```bash
# Start pretraining with default configuration
python scripts/train.py


# Standalone evaluator debugging
python -m modules.evaluator
```

### Environment Setup
- Activate environment: `conda activate py310` or equivalent
- Ensure PyTorch compatibility for mixed precision training
- Set `TOKENIZERS_PARALLELISM=false` (already configured in train.py)

## Architecture Overview

### Multi-Modal ECG-Text Contrastive Learning
This repository implements a CLIP-style contrastive learning framework for ECG signals paired with Chinese medical reports:

```
ECG Signals (12-lead, 5000 points) → Segmentation → Image Conversion (GAF/RP/MTF) → TimeSformer → Embeddings
Text Reports (Chinese) → MedCPT Encoder (partially frozen) → Projection → Embeddings → Contrastive Loss
```

### Key Components Integration

**Model Architecture** (`modules/model.py`):
- `ECGPretrainModel`: Main model combining ECG and text encoders through CLIP framework
- ECG encoder: TimeSformer with spatiotemporal attention for 50-frame image sequences
- Text encoder: MedCPT-based encoder with layer-wise freezing (first 6 layers frozen)
- CLIP head: Learnable temperature parameter for contrastive loss scaling

**Data Processing Pipeline** (`modules/dataset.py`):
- Dual dataset modes: Pre-computed images vs runtime conversion
- ECG segmentation: 100-point segments with 100-point stride (50 segments per lead)
- Multi-modal conversion: Each segment → GAF, RP, MTF images (100×100 each)
- Channel reorganization: `(T, 12, 3, H, W)` → `(T, 36, H, W)` preserving lead identity

**Training Framework** (`modules/trainer.py`):
- Contrastive learning with bidirectional retrieval metrics
- Mixed precision training with gradient clipping (max_norm=1.0)
- Advanced scheduling: Linear warmup + cosine annealing restarts
- Hardware-aware dataset selection based on GPU model detection

### Configuration and Paths

**Path Management** (`configs/config.py`):
- Hierarchical path construction from project root to dataset locations
- Dataset root: `../../datasets` (relative to project structure)
- Checkpoints: `project_root/checkpoints_pretrain`（与 `code/` 同级）
- Hardware adaptation: Automatic switching between dataset types based on GPU

**Key Hyperparameters**:
- `batch_size: 5` - Small due to memory-intensive image conversion
- `lr_rate: 1e-3` - Higher learning rate for transformer training
- `weight_decay: 1e-5` - Standard regularization for transformers
- Early stopping based on Mean Recall (contrastive accuracy)

## Development Patterns

### Dataset Selection Logic
The training script automatically selects dataset mode based on GPU hardware:
```python
# In scripts/train.py:67
if "5070" in torch.cuda.get_device_name(0):
    # Use limited dataset for memory-constrained GPUs
    dataset = ECGDataset_pretrain_local(config, mode="train")
else:
    # Use full pre-computed dataset
    dataset = ECGDataset_pretrain(config, mode="train")
```

### Model Customization Points
- **ECG Encoders**: TimeSformer (default), ViT1D, ResNet1D, TimesNet available in `modules/layers/`
- **Text Encoders**: Currently MedCPT, but framework supports other medical domain models
- **Image Conversion**: GAF, RP, MTF methods configurable in dataset pipeline

### Evaluation Metrics
The trainer tracks comprehensive contrastive learning metrics:
- `text_to_image_top1/top5`: Text→ECG retrieval accuracy
- `image_to_text_top1/top5`: ECG→Text retrieval accuracy
- `mean_recall`: Primary metric for model selection (average of bidirectional top-1)
- `clip/temperature`: Learnable temperature parameter evolution

### Wandb Integration
- Project name: `ecg-pretrain` (configurable via `Config.wandb_project`)
- Automatic logging of training metrics, retrieval accuracies, and temperature
- Model watching with `log_freq=100` for gradient and parameter tracking

## Code Organization Patterns

### Module Dependencies
- `sys.path` manipulation in multiple files for relative imports
- Circular import avoidance through careful import ordering
- Layer dependencies: `timesformer_pytorch` and `x_clip` require special path handling

### Memory Optimization
- Vectorized ECG processing using NumPy operations
- Persistent workers and pin_memory in DataLoader for efficiency
- Mixed precision training to reduce memory footprint
- Gradient clipping for training stability

### Extensibility Design
- Modular encoder architecture allows easy swapping of components
- Configuration-driven hyperparameter management
- Framework supports additional downstream tasks beyond pretraining
- Hardware-aware adaptations for heterogeneous training environments

## Important Implementation Details

### ECG Processing Pipeline
The conversion from raw ECG to multi-channel images is a critical architectural decision:
1. Raw 12-lead signals (5000 points each) → segmented into 100-point windows
2. Each window converted to 3 complementary image representations
3. Images organized as 36-channel sequence preserving lead and modality information
4. TimeSformer processes this as spatiotemporal data for rich feature extraction

### Medical Domain Considerations
- Text encoder specifically chosen for medical domain (MedCPT)
- Chinese medical reports processed through domain-aware tokenization
- Layer-wise freezing strategy balances domain adaptation with knowledge retention

### Hardware Adaptation Strategy
The codebase includes intelligent hardware detection for optimal performance:
- Memory-constrained GPUs (RTX 5070) use runtime ECG conversion with limited data
- Higher-end GPUs use pre-computed images for faster training
- Automatic fallback ensures training works across different hardware configurations
