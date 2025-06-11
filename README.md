# PlantCTCIP

**PlantCTCIP** is a deep learning model that integrates Convolutional Neural Networks (CNN) and Transformer architectures for accurate chromatin interaction prediction in multiple plants. This innovative approach combines the spatial feature extraction capabilities of CNNs with the sequential modeling power of Transformers to predict chromatin interactions across different plant species.


## PlantCTCIP Introduction

We have developed a deep learning model named PlantCTCIP for predicting chromatin interactions across multiple crop species. This model adopts a hybrid architecture: it first employs convolutional layers to extract spatial features from genomic data, and then uses Transformer encoder layers to capture long-range dependencies and complex interaction patterns. This approach enables accurate prediction of chromatin interactions, which is crucial for understanding plant genome architecture and gene regulatory mechanisms.


The PlantCTCIP model consists of three main components:
- **Convolutional Feature Extractor**: Multi-layer CNN with batch normalization and max pooling for spatial feature extraction
- **Transformer Encoder**: Multi-head attention mechanism with 6 encoder layers for capturing sequential dependencies
- **Fully Connected Classifier**: Deep neural network for final interaction prediction

## Data Introduction

The model is trained and evaluated on chromosome interaction data from multiple plant species, including:
- **PPIs: Promoter proximal region interaction
- **PDIs: Promoter-proximal and distal interactions
- **GWI: Whole genome chromatin interaction
- **Four Species: The training data includes four crops: maize, rice, wheat, and cotton. The data used includes: reference genome, chromatin interactions, expression levels, TFs, eQTLs, and multi omics data.

## Environment

### CUDA Environment

If you are running this project using GPU, please configure CUDA and cuDNN according to this version.

|       | Version |
| :---- | ------- |
| CUDA  | 11.1    |
| cuDNN | 8.0.5   |

### Package Environment

See "requirements_condalist.txt" for specific details.

Key dependencies include:
- Python 3.6.13
- PyTorch 1.8.1 / 1.10.2+cu113
- NumPy 1.19.5
- Pandas 1.1.5
- Scikit-learn 0.24.2
- Matplotlib 3.3.4
- Biopython 1.78 (Bio.Seq)
- H5py 3.1.0

## Model Architecture

The PlantCTCIP model architecture includes:

- **Input Layer**: Accepts single-channel genomic sequence data
- **Convolutional Layers**: 
  - 6 convolutional layers with increasing then decreasing channel dimensions (1→64→128→256→128→64→4)
  - Batch normalization and ReLU activation after each layer
  - Max pooling operations for dimensionality reduction
- **Transformer Encoder**:
  - 6 transformer encoder layers
  - 4-head multi-head attention
  - Model dimension: 4, Feed-forward dimension: 8
- **Classification Head**:
  - Fully connected layers with dropout (0.3)
  - Final output: 2 classes (interaction/no interaction)

## Training

The model is trained using standard supervised learning with:
- Loss function: Cross-entropy loss
- Optimizer: Adam optimizer


## Questions

If you have any questions, we kindly invite you to contact us at wangzhenye@henau.edu.cn and liujianxiao@mail.hzau.edu.cn

