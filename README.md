# Predictive Modeling with Neural Networks

This repository contains a collection of predictive models implemented in Python. It explores the use of different neural network architectures, including Convolutional Neural Networks (CNNs) and Long Short-Term Memory (LSTMs), for various forecasting and classification tasks.

## Description

The primary goal of this project is to demonstrate the implementation and application of neural networks for solving real-world problems. This includes a from-scratch implementation of a CNN for image processing and several models designed for business cost forecasting.


## Features

This repository includes the following models:

### 1. Convolutional Neural Network (CNN) for Image Processing
* **Implementation:** A multi-layer CNN built from scratch using only the NumPy library.
* **Layers:** Includes manually implemented layers for:
    * `Convolution_2d`
    * `ReLU` (Rectified Linear Unit) Activation
    * `Pooling` (Max Pooling)
* **Functionality:** Processes an image through sequential CNN, ReLU, and Pooling layers to extract features, with visualizations for each layer's output.

### 2. ANN & LSTM for Cost Forecasting
* **Models:** Implementations of Artificial Neural Networks (ANN) and Long Short-Term Memory (LSTM) networks.
* **Applications:** These models are designed to forecast various business-related costs, including:
    * Sales and Storing Costs (`annLSTMsalesstoringcost`)
    * Packaging Costs (`LSTMpackagingcost`)
    * Raw Material Costs (`LSTMrawmaterial`)
* **Testing:** Includes corresponding scripts to test the models' performance.


## Technologies Used

* **Language:** Python
* **Libraries:**
    * **NumPy:** For all numerical operations and core model implementation.
    * **Matplotlib:** For visualizing the output of network layers.
    * **Pillow (PIL):** For opening and processing images.


## How to Use

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    ```

2.  **Install the dependencies:**
    ```bash
    pip install numpy matplotlib pillow
    ```

3.  **Run a script:**
    To run the CNN image processing example, ensure you have an image file (e.g., `eagle_grayscale.jpeg`) in the correct path as specified in the script.
    ```bash
    python CNN_More_complex_example.py
    ```
    To run the cost forecasting models, execute the desired Python script:
    ```bash
    python annLSTMsalesstoringcost.py
    ```

