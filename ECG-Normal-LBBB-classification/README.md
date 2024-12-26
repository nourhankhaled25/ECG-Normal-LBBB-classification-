
# ECG-Normal-LBBB-classification

 provides a framework for classifying Electrocardiogram (ECG)
## Run Locally

Clone the project first

```bash
  https://github.com/Mohammed-Gadd/ECG-Normal-LBBB-classification.git
```
Install streamlit if not installed

```bash
  pip install streamlit
```

## Features

- **Data Preprocessing**:

    * Mean removal
    * Bandpass filter using a Butterworth filter (0.5 to 40 Hz).
    * Normalization to the range [-1, 1].
 
* **Feature Extraction**:

    * Wavelet transform with level 2 decomposition using Daubechies (DB3) wavelets.
 
- **Machine Learning Models**:

    - Support Vector Machines (SVM)
    - k-Nearest Neighbors (KNN)
    - Random Forest
      
- **Visualization**:

    - Provides plots of ECG signals and classification results.


