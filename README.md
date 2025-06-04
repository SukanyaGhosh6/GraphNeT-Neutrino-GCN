## Description:

GraphNeT-Neutrino-GCN applies Graph Neural Networks to reconstruct neutrino events from sparse detector data, predicting origin and energy. Built with PyTorch Geometric, it supports data preprocessing, training, and result visualization.

---

##  README.md

### GraphNeT-Neutrino-GCN

**Graph Neural Network for Neutrino Event Reconstruction**

#### Overview

This project implements a GNN-based approach for identifying and reconstructing neutrino events in detector simulations. It uses spatial-temporal data captured in neutrino detectors and maps it into a graph structure to predict key physics properties.

#### Features

* Graph data structure for physics events
* Custom GNN architecture with PyTorch Geometric
* Reproducible experiments
* Event visualization tools
* Open research code and reproducible workflow

#### Installation

```bash
# Clone the repository
$ git clone https://github.com/SukanyaGhosh6/GraphNeT-Neutrino-GCN
$ cd GraphNeT-Neutrino-GCN

# Create a virtual environment
$ python -m venv venv
$ source venv/bin/activate

# Install dependencies
$ pip install -r requirements.txt
```

#### Usage

```bash
# Train the model
python src/train.py --config config.yaml

# Evaluate the model
python src/evaluate.py --model-checkpoint outputs/model_latest.pth

# Visualize events
python src/visualize.py --event-id 42
```

#### Folder Structure

```
GraphNeT-Neutrino-GCN/
├── data/
├── src/
├── models/
├── notebooks/
├── config.yaml
└── README.md
```

#### License

MIT License

