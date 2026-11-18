
# Predictive Modelling

### Neural Network Forecasting, Monte Carlo Simulation, and Model Predictive Control

<p align="center">

![Python](https://img.shields.io/badge/Python-3-blue?logo=python&logoColor=white)
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-ANN%20%7C%20LSTM-purple)
![Framework](https://img.shields.io/badge/Framework-TensorFlow%20%7C%20PyTorch-orange)
![Simulation](https://img.shields.io/badge/Simulation-Monte%20Carlo%20%7C%20MPC-green)
![Research](https://img.shields.io/badge/Project-Predictive%20Modelling-red)

</p>

## 1. Overview

Predictive Modelling is a collection of Python-based implementations exploring neural network forecasting, stochastic simulation, and model-based control.

The repository brings together different approaches to predictive modeling, including Artificial Neural Networks (ANNs), Long Short-Term Memory (LSTM) networks, Monte Carlo simulation, and Model Predictive Control (MPC).

The implementations demonstrate how machine learning, deep learning, and mathematical optimization can be applied to business forecasting, asset-price scenario simulation, and autonomous control environments.

The project is organized into three primary areas:

- **Neural Network Forecasting:** ANN and LSTM models for business-related cost and quantity forecasting.
- **Monte Carlo Simulation:** Randomized asset-price scenarios for analyzing possible changes in asset values.
- **Model Predictive Control:** Neural-network-based environment modeling and trajectory optimization for control tasks.

## 2. Key Components

### 2.1 Neural Network Forecasting (ANN & LSTM)

This component explores Artificial Neural Networks and Long Short-Term Memory networks for forecasting business-related variables.

The implementations use historical data, normalization, lagged observations, neural network training, and prediction evaluation.

**Implemented forecasting tasks:**

| Application | Model | Python Script |
|---|---|---|
| Average count forecasting | ANN + LSTM | `annLSTMavcount.py` |
| Sales and storing cost forecasting | ANN + LSTM | `annLSTMsalesstoringcost.py` |
| Packaging cost forecasting | LSTM | `LSTMpackingcost.py` |
| Raw material cost forecasting | LSTM | `LSTMrawmaterial.py` |
| Average count prediction | ANN (MLP) | `anntestavcount.py` |
| Sales and storing cost prediction | ANN (MLP) | `anntestsalesstoringcost.py` |

**Technical characteristics:**

- Dense neural network layers.
- LSTM recurrent layers for sequential data.
- Min-Max normalization.
- Lag-based time-series preparation.
- Training and testing data separation.
- Mean Squared Error (MSE) evaluation.
- Root Mean Squared Error (RMSE) evaluation.
- Matplotlib-based prediction visualization.

**Main libraries:**

- NumPy
- Pandas
- TensorFlow / Keras
- Scikit-learn
- Matplotlib
- Seaborn

### 2.2 Monte Carlo Asset Simulation

The Monte Carlo component explores possible asset-price trajectories through randomized simulation.

It includes a Bitcoin-oriented implementation that generates multiple possible asset-value paths using randomized upward and downward price changes.

**Main implementation:**

`MC-prediction/montecarlobitcoin.py`

**Features:**

- Monte Carlo-based scenario simulation.
- Multiple randomized price trajectories.
- Historical price data input.
- Current price retrieval through an external API.
- Configurable simulation horizon.
- Configurable number of simulation paths.
- Visualization of simulated asset values.
- Optional plot export.

**Simulation parameters:**

| Parameter | Description |
|---|---|
| `n` | Simulation horizon |
| `x` | Number of simulation paths |
| `f` | Simulation density factor |
| `dailyChange` | Configurable daily percentage-change input |
| `savePlot` | Enables or disables plot saving |

**Important:** This component is intended for stochastic scenario exploration, not reliable financial forecasting. Its randomized paths should not be interpreted as actual future-price predictions or investment recommendations.

### 2.3 Model Predictive Control (MPC)

The Model Predictive Control component explores model-based control using neural networks and trajectory optimization.

It implements a learned environment model that predicts future states and rewards based on current states and actions.

The controller uses these predictions to evaluate possible future action sequences and select an action intended to maximize cumulative reward.

**Main components:**

| File | Responsibility |
|---|---|
| `main.py` | Initializes the environment, model, trainer, and MPC controller |
| `model.py` | Neural network environment models and model training |
| `mpc.py` | Model Predictive Control algorithm |
| `control.py` | Trajectory optimization and Cross-Entropy Method (CEM) |
| `diagnose.py` | Evaluates environment-model prediction errors |
| `test.py` | Contains an optimization test example |

**Core techniques:**

- Neural network environment modeling.
- State and action representation.
- Reward prediction.
- Model-based reinforcement learning.
- Model Predictive Control.
- Cross-Entropy Method (CEM) optimization.
- Trajectory planning.
- Prediction error diagnostics.

**Main libraries:**

- PyTorch
- NumPy
- OpenAI Gym

The current entry point configures a Pendulum environment.

## 3. Repository Structure

```text
predictive-modelling/
│
├── README.md
│
├── Neural Network Forecasting
│   │
│   ├── annLSTMavcount.py
│   ├── annLSTMsalesstoringcost.py
│   ├── LSTMpackingcost.py
│   ├── LSTMrawmaterial.py
│   ├── anntestavcount.py
│   └── anntestsalesstoringcost.py
│
├── MC-prediction/
│   │
│   ├── README.md
│   ├── montecarlobitcoin.py
│   │
│   ├── datasets/
│   │   └── market-price.csv
│   │
│   └── results/
│       └── Simulation result images
│
└── model predictive control/
    │
    ├── main.py
    ├── model.py
    ├── mpc.py
    ├── control.py
    ├── diagnose.py
    └── test.py
```

## 4. Technologies and Dependencies

The repository uses Python and several scientific computing, deep learning, and simulation libraries.

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| NumPy | Numerical computation |
| Pandas | Dataset processing |
| Matplotlib | Visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Data preprocessing and evaluation |
| TensorFlow / Keras | ANN and LSTM implementations |
| PyTorch | Neural network environment modeling |
| OpenAI Gym | Simulation environments |
| Requests | External price-data retrieval |

The MPC entry point also imports `quanser_robots`; install the appropriate package if required by the environment.

## 5. Installation

### 5.1 Clone the Repository

```bash
git clone https://github.com/Miraj-Rahman-AI/predictive-modelling.git

cd predictive-modelling
```

### 5.2 Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### 5.3 Install Dependencies

```bash
python -m pip install --upgrade pip

pip install numpy pandas matplotlib seaborn scikit-learn tensorflow requests torch gym
```

Additional environment-specific dependencies may be required for the MPC implementation.

## 6. Running the Experiments

### 6.1 ANN and LSTM Forecasting

The forecasting scripts expect their corresponding CSV datasets in the repository root.

| Python Script | Required Dataset |
|---|---|
| `annLSTMavcount.py` | `tata_av_count.csv` |
| `anntestavcount.py` | `tata_av_count.csv` |
| `annLSTMsalesstoringcost.py` | `tata_total_sales_storing_cost.csv` |
| `anntestsalesstoringcost.py` | `tata_total_sales_storing_cost.csv` |
| `LSTMpackingcost.py` | `tata_packing_cost.csv` |
| `LSTMrawmaterial.py` | `tata-raw_material.csv` |

These business datasets are not included in the current repository tree. Provide the corresponding files before executing the scripts.

Example:

```bash
python annLSTMavcount.py
```

Other examples:

```bash
python anntestavcount.py

python annLSTMsalesstoringcost.py

python LSTMpackingcost.py

python LSTMrawmaterial.py
```

### 6.2 Monte Carlo Simulation

Navigate to the Monte Carlo directory:

```bash
cd MC-prediction
```

Before running, update the `csvFile` variable in `montecarlobitcoin.py` to match the dataset included in the repository:

```python
csvFile = "datasets/market-price.csv"
```

Create the output directory if it does not exist:

```bash
mkdir -p IMG
```

Run the simulation:

```bash
python montecarlobitcoin.py
```

For verbose execution:

```bash
python montecarlobitcoin.py -v
```

The simulation generates randomized asset-price trajectories and can save the resulting visualization.

### 6.3 Model Predictive Control

Navigate to the MPC directory:

```bash
cd "model predictive control"
```

Run the main implementation:

```bash
python main.py
```

The main program initializes the Pendulum environment, trains the neural network environment model, and executes the MPC learning process.

**Compatibility note:** The current implementation uses the older Gym environment interface and `Pendulum-v0`. Newer Gym and Gymnasium versions may require API and environment-name updates.

## 7. Evaluation and Visualization

The repository includes several approaches to evaluating model behavior.

**ANN and LSTM:**

- Training and testing loss.
- Mean Squared Error (MSE).
- Root Mean Squared Error (RMSE).
- Observed versus predicted values.
- Prediction-quality plots.

**Monte Carlo:**

- Multiple simulated asset-price paths.
- Randomized asset-value trajectories.
- Graphical comparison of possible scenarios.

**Model Predictive Control:**

- Environment-model dynamics prediction error.
- Reward prediction error.
- Cumulative episode reward.
- Trajectory optimization behavior.

## 8. Current Scope and Limitations

The repository is a collection of experimental implementations rather than a unified production-ready predictive modeling framework.

Some important considerations:

1. **Dataset availability:** The business forecasting scripts require CSV files that are not currently included in the repository.
2. **Time-series validation:** The existing ANN and LSTM scripts use shuffled train/test splits. Chronological evaluation and training-only scaler fitting should be considered before treating their metrics as reliable forecasting results.
3. **Monte Carlo assumptions:** Randomized simulation paths are illustrative scenarios and do not guarantee accurate future asset values.
4. **Monte Carlo data paths:** The simulation script's original CSV path differs from the location of the dataset currently committed to the repository.
5. **MPC compatibility:** The control implementation uses older Gym APIs and may require modifications for modern software environments.
6. **Reproducibility:** Dependency versions and execution environments are not yet pinned in a requirements or environment specification file.

## 9. Research and Learning Objectives

This project explores several important areas of predictive modeling:

- Artificial Neural Networks and deep learning.
- Recurrent neural networks and sequential data modeling.
- Time-series forecasting.
- Stochastic processes and Monte Carlo simulation.
- Neural network-based environment modeling.
- Model Predictive Control.
- Trajectory optimization and model-based reinforcement learning.

It provides a foundation for exploring how predictive models can support forecasting, simulation, and intelligent decision-making.

## 10. Author

**Mirage Rahman [何善]**

Research Fellow, College of AI, Tsinghua University

Research Interests:
- Artificial Intelligence
- Machine Learning and Deep Learning
- Neural Network Modeling
- Predictive Analytics
- AI Agents and Autonomous Systems

GitHub: https://github.com/Miraj-Rahman-AI

LinkedIn: https://www.linkedin.com/in/miraj-rahman-ai/

---

<p align="center">

**Predictive Modelling | Neural Networks | Simulation | Intelligent Control**

</p>
