ORBITRON: AI-Driven Mission Risk Intelligence
ORBITRON is a unified physics-aware and explainable AI framework designed to evaluate multi-domain space mission risks within a single, integrated platform. By combining machine learning, physics-based simulations, and risk aggregation, it replaces fragmented aerospace monitoring systems with a comprehensive decision-support tool.

Project Overview
Space missions are subject to complex, independent risks that are traditionally analyzed in silos. ORBITRON bridges these gaps by integrating four core analytical modules into a centralized Mission Intelligence Fusion Layer.

The primary output of the system is the Mission Risk Index (MRI), a unified metric that quantifies overall mission safety by aggregating predictions from different domains.

Key Performance Benchmarks
Launch Prediction: 0.948 AUC

Satellite Anomaly Detection: 0.91 F1-score

RUL Estimation: 11.7 cycles MAE

Core Modules
1. Launch Failure Prediction
This module utilizes structured mission parameters to predict the likelihood of a successful launch.

Models: XGBoost, Random Forest, Gradient Boosting, and a Soft Voting Ensemble.

Explainability: Uses SHAP (SHapley Additive exPlanations) to provide feature-level risk analysis and transparency.

2. Satellite Health Monitoring
Analyzes telemetry signals to identify abnormal satellite behavior and degradation patterns.

Techniques: Autoencoder-based anomaly detection and PCA (Principal Component Analysis) for high-dimensional data visualization.

Dataset: NASA CMAPSS FD001.

3. Remaining Useful Life (RUL) Prediction
Estimates the operational lifespan of spacecraft subsystems before a critical failure occurs.

Approach: Implements degradation modeling and rolling window feature extraction through a regression pipeline.

4. NEO Orbit Prediction
Simulates the trajectories of Near-Earth Objects to assess potential collision risks.

Methods: Keplerian propagation combined with Monte Carlo simulations to account for trajectory uncertainty.

Mission Intelligence Fusion Layer
The Fusion Layer acts as the central engine of ORBITRON. It processes outputs from the Launch, Satellite, and NEO modules, applying weighted aggregation to generate the Mission Risk Index (MRI). This index categorizes the overall mission status into distinct risk levels, enabling rapid decision-making.

System Architecture and Tech Stack
Workflow
User Input: Mission parameters and telemetry data.

Preprocessing: Data cleaning and feature engineering.

Module Analysis: Simultaneous processing across Launch, Satellite, and NEO engines.

Fusion: Aggregation of domain-specific risks into the MRI.

Visualization: Real-time reporting via the dashboard.

Technologies
Language: Python

Machine Learning: Scikit-learn, XGBoost, TensorFlow, Keras

Data & Visualization: Pandas, NumPy, Streamlit, Plotly, Matplotlib

Installation and Usage
Setup
Clone the repository:
git clone https://github.com/srirangambadrinath/ORBITRON

Enter the directory:
cd ORBITRON

Install dependencies:
pip install -r requirements.txt

Running the Platform
Launch the interactive dashboard:
streamlit run dashboard/app.py

Future Enhancements
Integration of real-time satellite telemetry streams.

Cloud-native deployment for scalable mission monitoring.

Application of Reinforcement Learning for mission path optimization.

Expansion to multi-satellite fleet intelligence.

Research Authors:


S. V. Badrinath

A. Sujitha

K. Devendra Kumar

S. Sirajuddeen

Department of Artificial Intelligence & Data Science, Miracle Educational Society Group of Institutions.
