
# ETHOS: Efficient Transformers via Hypernetwork-Organized Sparsity

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

This repository contains the implementation of ETHOS from the paper "Why Play the Lottery When You Can Just Win?"

ETHOS is a novel architecture that dynamically generates millions of tiny experts from compressed latent representations, achieving 8.7B parameter capacity while using ~20× fewer FLOPs.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ethos.git
cd ethos

# Install dependencies (requires Python 3.10+, CUDA 11.8+)
pip install -r requirements.txt
```

## Quick Start

### Training ETHOS

The simplest way to train ETHOS is:

```bash
python train.py
```

This will:
1. Download 1% of the C4 dataset (configurable)
2. Train for 3 epochs with default settings
3. Save checkpoints and training logs

To use a custom configuration:

```bash
python train.py --config configs/default.yaml
```

### Model Architecture

ETHOS combines several key innovations:
- **Dynamic expert generation**: Instead of storing millions of expert parameters, we generate them from 128-dimensional latent codes
- **Product-key routing**: Efficient O(√N) routing to 262K experts per layer utilizing Query BatchNorm from PEER
- **Reordered execution**: Custom Triton kernel achieving 8× speedup

### Repository Structure

```
ethos/
├── model.py          # All model components
├── data.py           # Data loading and tokenization  
├── train.py          # Training script
├── monitor.py        # Training visualization
├── kernels.py        # Triton kernel implementation
├── configs/          # Configuration files
└── notebooks/        # Demo notebooks
```

### Configuration

Key parameters in `configs/default.yaml`:
- `num_experts`: 262,144 (512²) experts per layer
- `d_latent`: 128-dimensional latent codes
- `top_k`: 16 experts selected per token
- `num_routing_heads`: 8 independent routing heads

### Monitoring Training

Training progress is automatically logged and visualized:
- Real-time plots of loss, perplexity, learning rate
- CSV and JSON logs saved to `training_logs/`
- Checkpoints saved to `checkpoints/`

### Requirements

- PyTorch 2.0+
- CUDA 11.8+
- Triton 2.1+
- Flash Attention 2.0+
- 80GB+ GPU memory recommended

## Paper Results

On 1% of C4 dataset:
- Perplexity: 34.85 after 4B tokens
- Training speed: 15K tokens/second on GH200
- Memory efficiency: 16× reduction vs PEER

## Citation

```bibtex
@article{medford2025ethos,
  title={Why Play the Lottery When You Can Just Win?},
  author={Medford, Wesley and McCormick, Chris and Callicoat, Eve},
  journal={arXiv preprint arXiv:2407.XXXXX},
  year={2025}
}
```

## License

This project is licensed under the **GNU Affero General Public License v3.0 (AGPLv3)** - see the [LICENSE](LICENSE) file for details.

**Important**: The AGPLv3 license requires that any modifications or derivative works be released under the same license, including when used as a network service.

### Commercial Licensing

For commercial use cases that require a different license, please contact **wryanmedford@gmail.com** to discuss commercial licensing options.
