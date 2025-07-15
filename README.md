# ETHOS: Efficient Transformers via Hypernetwork-Organized Sparsity

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

This repository contains the implementation of ETHOS from the paper "Why Play the Lottery When You Can Just Win?"

ETHOS is a novel architecture that dynamically generates millions of tiny experts from compressed latent representations, achieving 8.7B parameter capacity while using ~20× fewer FLOPs.

> **📜 License**: This project is licensed under AGPLv3. For commercial licensing options, contact wryanmedford@gmail.com

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ethos.git
cd ethos

# Install dependencies (requires Python 3.10+, CUDA 11.8+)
pip install -r requirements.txt