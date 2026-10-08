# Prosthetic AI: Live EMG Control System with Haptic Feedback & Continual Learning

## 📖 Project Overview
This project presents a medical-grade, edge-deployable AI system for controlling a prosthetic limb using High-Density Surface Electromyography (HD-sEMG) signals. 

Unlike traditional static models, this system utilizes **Continual Metric Learning** and **Streaming Linear Discriminant Analysis (SLDA)** to adapt to new users and new movements without catastrophic forgetting. Furthermore, it features a biologically inspired **Haptic Feedback Loop** using an Adaptive Leaky Integrate-and-Fire (ALIF) neuron, providing the user with a true sense of touch based on physical Force Sensitive Resistors (FSR).

## 🌟 Key Features
- **Metric Continual Learning:** Uses Supervised Contrastive Learning (SupCon) to build a robust, generalized embedding space for EMG signals.
- **Edge-Optimized Inference:** Replaces heavy neural classifiers with **Streaming LDA (SLDA)**, enabling ultra-fast, memory-efficient inference without backpropagation.
- **Medical-Grade Safety:** Implements **Monte Carlo (MC) Dropout** for Uncertainty Quantification. If the model is unsure, it safely defaults to the "Rest" state to prevent accidental movements.
- **Closed-Loop Haptic Feedback:** Uses an **ALIF Neuron** to convert physical grip force (from FSR sensors) into electrical spikes, simulating natural biological touch.
- **Domain Adaptation (UDA):** Employs **Deep CORAL** with an Anchor Loss to maintain performance even when sensors shift or the user sweats (Electrode Shift/Sweat simulation).
- **Biological Intent Detection:** Calculates the **Co-contraction Index** to detect if the user is trying to perform a fragile/precise movement.


🧠 Architecture Details
1. The Backbone (Metric_ContinualNet)
A hybrid neural network combining:
Dilated Multi-Scale Convolutions: To capture both high-frequency tremors and low-frequency muscle envelopes.
Sensor Attention Gates: To dynamically suppress noisy or displaced EMG channels.
GRU + Temporal Attention: To model the sequential dynamics of muscle contractions.
2. The Classifier (StreamingLDA)
Instead of a standard Softmax layer, we use SLDA. It incrementally updates class means and the covariance matrix. This allows the system to learn new gestures in a Few-Shot manner (using only 3-5 repetitions) without retraining the neural network.
3. Haptic Feedback (ALIF_Encoder)
A spiking neuron model that translates the analog voltage from the FSR (Force Sensitive Resistor) into discrete electrical pulses (Spikes), mimicking the mechanoreceptors in human skin.
📊 Performance Metrics
The system is evaluated on the NinaPro DB2 dataset across multiple subjects, focusing on:
Movement Accuracy: Precision in classifying dynamic gestures.
False Activation Rate: Crucial for prosthetics; ensuring the hand doesn't move during the "Rest" state.
Uncertainty Rejection Rate: How effectively the system identifies and rejects ambiguous signals.