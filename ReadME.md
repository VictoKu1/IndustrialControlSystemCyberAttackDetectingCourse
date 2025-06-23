# Industrial Control System Power - Cyber Attacks Detection 

###### Ariel University, Israel || Semester A, 2022

## [Presentation](https://www.canva.com/design/DAFbkwt6KjQ/7im1RNlKjYkdZpg1dugivg/view?utm_content=DAFbkwt6KjQ&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)

[![images_01](https://github.com/VictoKu1/IndustrialControlSystemCyberAttackDetectingCourse/blob/master/Media/1.gif)](https://www.canva.com/design/DAFbkwt6KjQ/7im1RNlKjYkdZpg1dugivg/view?utm_content=DAFbkwt6KjQ&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Datasets](#datasets)
- [Models](#models)
- [Web Interface](#web-interface)
- [Docker Usage](#docker-usage)
- [API Documentation](#api-documentation)
- [Contributing](#contributing)
- [License](#license)
- [References](#references)

## 🎯 Project Overview

This project implements a machine learning-based intrusion detection system for Industrial Control Systems (ICS) in power systems. The system can classify power system events into three categories:
- **Normal Operations**: Regular power system behavior
- **Natural Events**: Faults and maintenance activities
- **Cyber Attacks**: Malicious activities including data injection, command injection, and relay setting changes

The system uses multiple machine learning algorithms to detect and classify cyber attacks in real-time, providing critical security monitoring for industrial power control systems.

## ✨ Features

- **Multi-class Classification**: Supports binary (2-class), triple (3-class), and multi-class (37-class) classification
- **Real-time Detection**: Processes CSV data files for immediate threat assessment
- **Multiple ML Algorithms**: Implements Random Forest, Neural Networks, K-Nearest Neighbors, Decision Trees, and more
- **Web Interface**: User-friendly Flask-based web application for easy data upload and analysis
- **Docker Support**: Containerized deployment for easy setup and deployment
- **Feature Engineering**: Advanced preprocessing including feature selection and scaling
- **Comprehensive Evaluation**: Detailed performance metrics and confusion matrices




### Core Components:

1. **Data Preprocessing**: Handles missing values, outliers, and feature scaling
2. **Feature Engineering**: Creates domain-specific features for power system measurements
3. **Model Training**: Multiple ML algorithms with hyperparameter optimization
4. **Prediction Engine**: Real-time classification of new data
5. **Web Interface**: User-friendly interface for data upload and results display

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Git (for cloning the repository)

### Local Installation

1. **Clone the repository**:
```bash
git clone https://github.com/VictoKu1/IndustrialControlSystemCyberAttackDetectingCourse.git
cd IndustrialControlSystemCyberAttackDetectingCourse
```

2. **Upgrade pip**:
```bash
pip install --upgrade pip
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

### Required Dependencies

The project requires the following Python packages:
- `scikit-learn==1.5.0` - Machine learning algorithms
- `pickle-mixin` - Model serialization
- `matplotlib` - Data visualization
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `flask` - Web framework (for web interface)


### Expected Data Format

Your CSV file should contain power system measurements with columns including:
- Relay measurements (R1-PA:Z, R2-PA:Z, etc.)
- Voltage phase angles (R1-PA1:VH, R1-PA2:VH, etc.)
- Current phase angles (R1-PA4:IH, R1-PA5:IH, etc.)
- Voltage phase magnitudes (R1-PM1:V, R1-PM2:V, etc.)
- Current phase magnitudes (R1-PM4:I, R1-PM5:I, etc.)
- Marker column for classification labels

## 📊 Datasets

The project includes three different dataset configurations:

### 2-Class Dataset
- **Location**: `Class/binaryAllNaturalPlusNormalVsAttacks/`
- **Description**: 37 event scenarios grouped as either attack (28 events) or normal operations (9 events)
- **Use Case**: Binary classification for attack detection

### 3-Class Dataset
- **Location**: `Class/triple/`
- **Description**: 37 event scenarios grouped into 3 classes:
  - Attack events (28 events)
  - Natural events (8 events)
  - No events (1 event)
- **Use Case**: Triple classification for detailed event analysis

### Multi-class Dataset
- **Location**: `Class/multiclass/`
- **Description**: Each of the 37 event scenarios as its own class
- **Use Case**: Fine-grained classification for specific event identification

## 🤖 Models

The system implements multiple machine learning algorithms:

### Supported Algorithms
- **Random Forest Classifier**: Ensemble method with high accuracy
- **Neural Networks (MLP)**: Deep learning approach
- **K-Nearest Neighbors**: Distance-based classification
- **Decision Trees**: Interpretable tree-based model
- **Logistic Regression**: Linear classification
- **AdaBoost**: Boosting ensemble method
- **Gradient Boosting**: Advanced ensemble technique
- **Extra Trees**: Extremely randomized trees

### Model Performance
- **Binary Classification**: High accuracy for attack vs. normal detection
- **Multi-class Classification**: Detailed classification of specific attack types
- **Feature Selection**: RFECV (Recursive Feature Elimination with Cross-Validation) for optimal feature selection

## 🌐 Web Interface

The web interface provides a user-friendly way to interact with the detection system:

### Features
- **File Upload**: Drag-and-drop or click-to-upload CSV files
- **Real-time Processing**: Immediate analysis and classification
- **Results Display**: Clear presentation of classification results
- **Session Management**: Secure handling of uploaded data


## 🐳 Docker Usage

### Quick Start with Docker

1. **Build the Docker image**:
```bash
docker build -t attack_detection_ui
```

2. **Run the container**:
```bash
docker run -it attack_detection_ui
```

### Docker Configuration

The Dockerfile includes:
- Python 3.8 base image
- All required dependencies
- Automatic repository cloning
- Pre-configured execution environment

### Dataset URL Format

For Docker usage, datasets should be accessible via URL in the format:
```
https://raw.githubusercontent.com/VictoKu1/IndustrialControlSystemCyberAttackDetectingCourse/master/Class/binaryAllNaturalPlusNormalVsAttacks/data1.csv
```

## 🔌 API Documentation

### Main Detection Function

```python
def calculate(test_df):
    """
    Main function for cyber attack detection
    
    Args:
        test_df (pandas.DataFrame): Input data with power system measurements
        
    Returns:
        list: Classification results for each data point
    """
```

### Preprocessing Functions

```python
def preprocess(df):
    """Feature engineering for power system data"""
    
def vectorize_df(df):
    """Data scaling and encoding"""
    
def remove_irrelevant_features(df):
    """Clean and prepare data for modeling"""
```

## 🤝 Contributing

We welcome contributions to improve the project:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature-name`
3. **Make your changes** and add tests
4. **Commit your changes**: `git commit -am 'Add feature'`
5. **Push to the branch**: `git push origin feature-name`
6. **Submit a pull request**

### Development Setup

For development, install additional dependencies:
```bash
pip install jupyter notebook matplotlib seaborn
```

## 📄 License

This project is part of academic research at Ariel University, Israel. Please refer to the university's academic integrity policies for usage guidelines.

## 🔗 Links

### Datasets
- **Original Dataset Source**: https://sites.google.com/a/uah.edu/tommy-morris-uah/ics-data-sets


## 📚 References

### Power System Datasets (Dataset 1)

Uttam Adhikari, Shengyi Pan, and Tommy Morris in collaboration with Raymond Borges and Justin Beaver of Oak Ridge National Laboratories (ORNL) have created 3 datasets which include measurements related to electric transmission system normal, disturbance, control, cyber attack behaviors. Measurements in the dataset include synchrophasor measurements and data logs from Snort, a simulated control panel, and relays.

[README Description](http://www.google.com/url?q=http%3A%2F%2Fwww.ece.uah.edu%2F~thm0009%2Ficsdatasets%2FPowerSystem_Dataset_README.pdf&sa=D&sntz=1&usg=AOvVaw3t-soxdA-27GPUSRG1JP_Q)

The power system datasets have been used for multiple works related to power system cyber-attack classification.

* [Pan, S., Morris, T., Adhikari, U., Developing a Hybrid Intrusion Detection System Using Data Mining for Power Systems, IEEE Transactions on Smart Grid. doi: 10.1109/TSG.2015.2409775](http://www.google.com/url?q=http%3A%2F%2Fieeexplore.ieee.org%2Fstamp%2Fstamp.jsp%3Ftp%3D%26arnumber%3D7063234%26isnumber%3D5446437&sa=D&sntz=1&usg=AOvVaw06Q-fkYHriTfgJYieCBnJc)

* [Pan, S., Morris, T., Adhikari, U., Classification of Disturbances and Cyber-attacks in Power Systems Using Heterogeneous Time-synchronized Data, IEEE Transactions on Industrial Informatics. doi: 10.1109/TII.2015.2420951](http://www.google.com/url?q=http%3A%2F%2Fieeexplore.ieee.org%2Fstamp%2Fstamp.jsp%3Ftp%3D%26arnumber%3D7081776%26isnumber%3D4389054&sa=D&sntz=1&usg=AOvVaw21tCmn-MAAmkUzCRpflyv_)

* [Pan, S., Morris, T., Adhikari, U., A Specification-based Intrusion Detection Framework for Cyber-physical Environment in Electric Power System, International Journal of Network Security (IJNS), Vol.17, No.2, PP.174-188, March 2015.](http://www.google.com/url?q=http%3A%2F%2Fijns.jalaxy.com.tw%2Fcontents%2Fijns-v17-n2%2Fijns-2015-v17-n2-p174-188.pdf&sa=D&sntz=1&usg=AOvVaw3qkk5GcOIxcgHQesgQjr5w)

* [Beaver, J., Borges, R., Buckner, M., Morris, T., Adhikari, U., Pan, S., Machine Learning for Power System Disturbance and Cyber-attack Discrimination, Proceedings of the 7th International Symposium on Resilient Control Systems, August 19-21,2014, Denver, CO, USA.](https://www.google.com/url?q=https%3A%2F%2Fdoi.org%2F10.1109%2FISRCS.2014.6900095&sa=D&sntz=1&usg=AOvVaw3fR5r_1bSnchlVhDlEXHXE)

### Additional Articles

1. [Industrial Control System Traffic Datasets For Intrusion Detection Research](https://link.springer.com/content/pdf/10.1007/978-3-662-45355-1_5.pdf)

2. [Cyber-Attack Detection for Industrial Control System Monitoring with Support Vector Machine Based on Communication Profile](https://www.researchgate.net/profile/Ichiro-Koshijima/publication/318127445_Cyber-Attack_Detection_for_Industrial_Control_System_Monitoring_with_Support_Vector_Machine_Based_on_Communication_Profile/links/59f477b50f7e9b553ebbdeb6/Cyber-Attack-Detection-for-Industrial-Control-System-Monitoring-with-Support-Vector-Machine-Based-on-Communication-Profile.pdf)

3. [Efficient Cyber Attack Detection in Industrial Control Systems Using Lightweight Neural Networks and PCA](https://arxiv.org/pdf/1907.01216)

4. [Measuring the Risk of Cyber Attack in Industrial Control Systems](https://dora.dmu.ac.uk/bitstream/handle/2086/13839/ewic_icscsr2016_paper12.pdf?sequence=1)

5. [An Ensemble Deep Learning-Based Cyber-Attack Detection in Industrial Control System](https://ieeexplore.ieee.org/iel7/6287639/8948470/09086038.pdf)

---

**Note**: This project is designed for educational and research purposes. For production deployment in critical infrastructure, additional security measures and validation should be implemented




















































































