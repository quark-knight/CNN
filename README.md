
# CNN-based machine learning to classify lepton track images on a toy detector model

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)](https://tensorflow.org/)
[![ROOT](https://img.shields.io/badge/ROOT-6.x+-darkblue.svg)](https://root.cern/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Latest-blue.svg)](https://scikit-learn.org/)
![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)
<!--![License](https://img.shields.io/badge/License-Open%20Source-green.svg
[![Code style: black](https://img.shields.io/badge/Code%20style-PEP8-blue.svg)](https://www.python.org/dev/peps/pep-0008/)-->

This repository contains the code that I developed as a semester project student in experimental high-energy physics, with a focus on real-time trigger applications at the Large Hadron Collider (LHC) at CERN.
This is a complete implementation of a data generation pipeline, the model implementation of a convolutional neural network (CNN) system, along with the results for classifying ( or differentiating between) the leptons with different transverse momentum ($p_T$) in a toy detector model based on their smeared or displaced track images. 

The project demonstrates an end-to-end machine learning pipeline for particle physics, spanning from simulated event generation through
detector response modeling to neural network training and evaluation using TensorFlow/Keras.
The system achieves an ROC AUC score of 0.9277 on high-quality data, illustrating the efficacy of deep learning approaches for detector-based classification tasks.

----
## Scientific Motivation and Background

### Transverse Momentum Classification in Particle Physics
At the LHC, particles are produced in high-energy proton-proton collisions. One of the most important observables in particle physics is the transverse momentum 
($p_T$) of these particles—the component of momentum perpendicular to the beam axis. Efficient and accurate classification of particles based on their $p_T$ is crucial for:

1. **Real-time Event Filtering**: The LHC produces ~1 billion collisions per second, but only ~1,000 events can be recorded. Trigger systems must rapidly decide which events to keep based on high-$p_T$ particle signatures
2. **Signal Enhancement**: High- $p_T$ leptons are often signatures of rare, interesting physics phenomena (heavy boson decays, new particles)
3. **Background Suppression**: Low- $p_T$ objects are typically copious backgrounds that must be efficiently rejected

### Traditional vs. Deep Learning Approaches
Historically, particle identification and kinematic classification has relied on hand-crafted features and conventional machine learning algorithms. This work explores whether convolutional neural networks, which have demonstrated remarkable success in image recognition, can learn effective representations directly from detector outputs represented as images.


## System Architecture

### Project Pipeline

The analysis follows a three-stage pipeline:

```
Event Generation (C++) → Image Generation (ROOT/Python) → Network Training (TensorFlow/Keras)
```

#### Stage 1: Event Generation
**Location**: `Event_Generation/`

Simulated particle events are generated using ROOT's C++ environment. The event generator:
- Produces lepton tracks within a specified range of kinematic properties (but random)
- Generates both high- $p_T$ (signal) and low- $p_T$ (background) tracks with configurable proportions
- Applies spatial resolution effects (smearing) to simulate detector response
- Can output track parameters in both Cartesian (x,y) and cylindrical (r,φ) coordinate systems

**Key Parameters**:
- Number of tracks per event
- Signal (high- $p_T$) vs. background (low- $p_T$) fractions
- Bool to turn on or off smearing for simulating detector resolution
- The amount of smearing for the hits
- Total number of events to generate


-----
#### Stage 2: Image Generation and Visualization
**Location**: `Image_Generation/`

Track data is converted into 2D detector images:
1. **Data Format Conversion**: Event data (text format) is converted to ROOT TTrees for efficient processing
2. **Image Rendering**: Track hits are plotted on 2D grids (typically 128×128 pixels) representing detector coordinates
3. **Image Classification**: Images are automatically sorted into HiPt (signal) and LoPt (background) directories
4. **Data Augmentation**: Training/testing sets are created with stratified splitting

The images directly represent detector readouts where:
- X and Y axes correspond to spatial detector coordinates
- Pixel intensity encodes hit multiplicity or position information
- Track characteristics are encoded in the spatial pattern of deposits or hits on the detector

<div align="center">
<table style="border-collapse: collapse; margin: 20px auto;">
<tr>
<th style="padding: 10px;"> Training data generation pipeline </th>
</tr>
<tr>
<td style="padding: 15px; text-align: center;">
<img src="https://github.com/user-attachments/assets/5a46118d-bd6e-4e4a-afc3-96dcaa07f624" alt=" Training data generation pipeline" width="900" />
</td>
</tr>
</table>
</div>

<!--<img width="1748" height="591" alt="image" src="https://github.com/user-attachments/assets/5a46118d-bd6e-4e4a-afc3-96dcaa07f624" />-->

<!--#### Sample tracks-->
<div align="center">
<table style="border-collapse: collapse; margin: 20px auto;">
<tr>
<th colspan="2 style="padding: 10px;"> Sample tracks </th>
<!-- <th style="padding: 10px;">With smearing </th> -->
</tr>
<tr>
<td style="padding: 15px; text-align: center;">
<img src="https://github.com/user-attachments/assets/15c2b4d5-c031-431e-b19e-c6de5a558c52" alt="Without smearing" width="250" />
</td>
<td style="padding: 15px; text-align: center;">
<img src="https://github.com/user-attachments/assets/1787b914-565f-45b6-96ae-1a0f7c6bce89" alt="With smearing" width="250" />
</td>
</tr>
<tr>
<td style="text-align: center;"><i>Example of track without smearing</i></td>
<td style="text-align: center;"><i>Example of track with smearing</i></td>
</tr>
</table>
</div>

<!--### Without smearing 
<img width="307" height="298" alt="image" src="https://github.com/user-attachments/assets/15c2b4d5-c031-431e-b19e-c6de5a558c52" />

### Without smearing 
<img width="417" height="404" alt="image" src="https://github.com/user-attachments/assets/1787b914-565f-45b6-96ae-1a0f7c6bce89" /> -->
----

#### Stage 3: Convolutional Neural Network
**Location**: `Network/`

A custom CNN architecture is trained on the generated images:

**Architecture**:
```
Input (128×128 grayscale) 
  ↓
Conv2D(32, 5×5) + ReLU + MaxPool(2×2) + BatchNorm
  ↓
Conv2D(64, 3×3) + ReLU + MaxPool(2×2) + BatchNorm
  ↓
Conv2D(64, 1×1) + ReLU + MaxPool(2×2) + BatchNorm
  ↓
Flatten
  ↓
Dense(128) + ReLU
  ↓
Dense(32) + ReLU
  ↓
Dense(1) + Sigmoid → [0,1] (High-pT probability)
```

**Training Configuration**:
- **Loss Function**: Binary cross-entropy
- **Optimizer**: Adam
- **Batch Size**: 32
- **Epochs**: 10
- **Validation Strategy**: Hold-out test set with 50/50 train/test split
- **Data Augmentation**: Horizontal flipping of training images for robustness
- **Regularization**: Batch normalization, dropout considerations

**Callbacks**:
- Model Checkpoint: Saves best model based on validation accuracy
- Optional: ReduceLROnPlateau for adaptive learning rate scheduling

-----

## Key Results

### Performance Metrics

The trained model achieves strong classification performance:

- **ROC AUC Score**: 0.9277 (excellent discrimination)
- **Test Accuracy**: ~77-84% depending on smearing configuration
- **True Positive Rate at 90% FPR**: ~95%
- **Model Convergence**: Stable within 10 epochs without divergence

<div align="center">
<table style="border-collapse: collapse; margin: 20px auto;">
<tr>
<th style="padding: 10px;">Classification score distributions </th>
<th style="padding: 10px;">ROC AUC curve </th>
</tr>
<tr>
<td style="padding: 15px; text-align: center;">
<img src="Result Images/1.png" alt="Classification score distributions" width="400" />
</td>
<td style="padding: 15px; text-align: center;">
<img src="Result Images/2.png" alt="ROC AUC curve" width="400" />
</td>
</tr>
<tr>
<td style="text-align: center;"><i>Reveals clear separation between signal & background scores </i></td>
<td style="text-align: center;"><i>Quantifies true positive vs. false positive trade-offs</i></td>
</tr>
</table>
</div>


<div align="center">
<table style="border-collapse: collapse; margin: 20px auto;">
<tr>
<th style="padding: 10px;">Accuracy vs. Training Epoch </th>
<th style="padding: 10px;">Loss Progression </th>
</tr>
<tr>
<td style="padding: 15px; text-align: center;">
<img src="Result Images/3.png" alt="Accuracy vs. Training Epoch" width="400" />
</td>
<td style="padding: 25px; text-align: center;">
<img src="Result Images/4.png" alt="Loss Progression" width="400" />
</td>
</tr>
<tr>
<td style="text-align: center;"><i>Demonstrates model learning progress and validation stability</i></td>
<td style="text-align: center;"><i>Shows well-behaved convergence with minimal overfitting</i></td>
</tr>
</table>
</div>


## Repository Structure

```
CNN-main/
├── README.md                          # This file
├── Installing packages.txt            # Dependency installation guide
├── Event_Generation/                  # Stage 1: Particle event simulation
│   ├── event_data_corr.C             # Main event generation script
│   ├── event_data.C                   # Alternative (old) event generator
│   └── README.txt                     # Detailed execution instructions
├── Image_Generation/                  # Stage 2: Image creation
│   ├── generate_command.py           # Utility for generating makeMyTree.C
│   ├── makeMyTree.C                   # ROOT script: TTree creation
│   ├── plotmaker.C                    # ROOT script: Image generation
│   ├── plotmaker2.C                   # Alternative (old) plotting script
│   ├── csvmaker.C                     # CSV export utility
│   └── README.txt                     # Detailed execution instructions
├── Network/                           # Stage 3: CNN training
│   ├── mycnn.py                       # Primary CNN training script
│   ├── testcnn.py                     # Model testing and evaluation
│   ├── track_cnn_test.py             # Inference/testing utilities
│   ├── oldcnn.py                      # old CNN script file
│   ├── TestingNNModel_Notebook.ipynb # Jupyter notebook for analysis
│   └── Basic Python/                  # Tutorial notebooks
│       ├── basic.ipynb
│       ├── matplotlib_basic.ipynb
│       ├── pandas_basic.ipynb
│       └── p.py
└── Result Images/                     # Output directory for plots (also, has plots for configs that didn't work)
```
## Installation and Setup

### Requirements

The project requires the following software and libraries:

**System Requirements**:
- ROOT 6.x or higher (installation: https://root.cern/)
- Python 3.7+
- C++ compiler (for ROOT)

>
> Please note that only specific numpy, tensorflow, and root versions work together for particular NVIDIA dependencies. So these versions are just indicative  
>

**Python Dependencies**:
```
numpy, matplotlib, scipy, scikit-learn ,tensorflow >= 2.0, keras, pandas
```
### Installation Instructions

#### 1. Install ROOT (if not already installed)
```bash
# Visit https://root.cern/install/ for detailed instructions
# On Linux distributions:
apt-get install root-system  # Ubuntu/Debian
conda install root           # Recommended to create a new environment and do the installation there
```
>
> **More recommended**- To build root from source (that's what I did), it makes the whole processs more stable 
>

#### 2. Install Python Dependencies
```bash
pip install tensorflow keras numpy matplotlib scikit-learn pandas scipy
```

Or use the provided requirements (create one from `Installing packages.txt`):
```bash
pip install -r requirements.txt
```

#### 3. Configure ROOT Environment
```bash
# Source ROOT initialization script (path depends on installation)
source /path/to/root/bin/thisroot.sh
```

------

## Execution Workflow - Complete Execution Pipeline 

Follow these steps in order to reproduce the full analysis:

#### Step 1: Generate Simulation Events
```bash
cd Event_Generation/
root -q -l -b 'event_data_corr.C(50,1,20000,"event_data_hipt.txt")'
root -q -l -b 'event_data_corr.C(50,0,20000,"event_data_lopt.txt")'
cd ..
```

**Parameters for `event_data_corr.C(nTracks, signalFraction, nEvents, outputFile)`**:
- `nTracks`: Hits per event (50 recommended)
- `signalFraction`: 1 for high-pT, 0 for low-pT
- `nEvents`: Total events to generate (20,000 for training)
- `outputFile`: Output text file name

#### Step 2: Create ROOT Datasets
```bash
cd Image_Generation/

# Configure and run the tree maker
python generate_command.py  # Edit generate_command.py to set nTracks

# Create ROOT files
root -q -l -b 'makeMyTree.C("event_data_hipt.txt","tree_hipt.root")'
root -q -l -b 'makeMyTree.C("event_data_lopt.txt","tree_lopt.root")'
cd ..
```

#### Step 3: Generate Training Images
```bash
cd Image_Generation/

# For 50 tracks and 16 layers: N = 50*16 = 800
root -q -l -b 'plotmaker2.C(800,"tree_hipt.root")'
root -q -l -b 'plotmaker2.C(800,"tree_lopt.root")'

# Images are created in: ./images/Training/ and ./images/Testing/
cd ..
```

#### Step 4: Train the CNN Model
```bash
cd Network/
python mycnn.py
```

**Output files generated**:
- `my_model.h5`: Trained CNN model weights
- `best_model.h5`: Best checkpoint during training
- `acc_v_epoch.png`: Accuracy vs. epoch plot
- `loss_v_epoch.png`: Loss vs. epoch plot

#### Step 5: Evaluate and Test
```bash
# Test the trained model
python testcnn.py

# Or use the Jupyter notebook for interactive analysis
jupyter notebook TestingNNModel_Notebook.ipynb
```

## References and Further Reading

### High-Energy Physics Background
- LHC Physics: https://home.cern/science/physics/large-hadron-collider
- Trigger Systems: https://atlas.cern/updates/atlas-blog/trigger-and-data-acquisition
- Track Reconstruction: https://arxiv.org/abs/1905.03475

### Machine Learning in Particle Physics
- ML4Jets: https://www.mlphysics.org/
- IML Topical Group: https://usatlas.bnl.gov/physics/ML/

### Tool Documentation
- ROOT: https://root.cern/
- TensorFlow/Keras: https://tensorflow.org/
- scikit-learn: https://scikit-learn.org/

## Future directions 
- Extend the model to generate hits in a 3D detector
- Train the model on the 3D detector
- Translate the results to trigger applications

## Contact and Support

For questions, bug reports, or feature requests, please contact:
- Author: Aditya Shukla
- Email: aditya.shukla@students.iiserpune.ac.in
- Institution: IISER Pune

## Acknowledgments

This work was developed as part of a semester project in experimental high-energy physics, with guidance from Prof. Sourabh Dube, at IISER Pune.. The project benefited from the rich ecosystem of open-source tools in particle physics (ROOT, Geant4) and machine learning (TensorFlow, scikit-learn).

---

**Last Updated**: March 2026
**Version**: 2.0

