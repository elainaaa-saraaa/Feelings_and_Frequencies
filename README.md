>Feelings & Frequencies

Emotion Recognition from Speech Using Classical ML and Signal Processing

>Overview

Feelings & Frequencies is a speech‑emotion recognition system that blends Digital Signal Processing (DSP) with Classical Machine Learning to classify human emotions from a>udio.
Instead of relying on deep learning, this project focuses on feature‑engineered audio analysis, making it lightweight, interpretable, and grounded in core ECE concepts.
The system extracts spectral features from speech, trains a Random Forest classifier, visualizes emotion clusters using PCA, and evaluates robustness under real‑world distortions such as noise, pitch shifts, speed changes, and volume variations.

> Key Features
1. Digital Signal Processing (DSP) Feature Extraction
   
a. Zero Crossing Rate (ZCR)

b. Spectral Centroid

c. Spectral Bandwidth

d. Spectral Rolloff

These features capture the energy, brightness, sharpness, and frequency spread of speech — all of which vary with emotion.

2. Classical Machine Learning
   
a. Random Forest classifier

b. Train/test split with stratification

c. Classification report + accuracy metrics

d. Probability‑based emotion prediction

3.  Visualization
   
a. PCA emotion clusters

b. Confusion matrix

c. Probability distribution bar chart


4. Robustness Testing
   
Simulates real‑world distortions:

- Additive noise
  
- Pitch shifting
  
- Speed changes
  
- Volume scaling
  
Each modified signal is re‑classified to evaluate model stability.

>Dataset

This project uses the RAVDESS dataset:

Ryerson Audio-Visual Database of Emotional Speech and Song.

You must download it manually due to licensing.



>Why this project matters?

Working on this project gave me a chance to bring together two things I genuinely enjoy: the structure and logic of signal processing, and the human side of emotion in speech. It was interesting to see how something as technical as spectral features could reveal something as personal and expressive as how someone feels. Building this system helped me understand speech not just as a waveform, but as a reflection of energy, tension, calmness, and personality. It also showed me how classical ML and DSP can still create meaningful, interpretable results without needing massive deep‑learning models. For me, this project sits at the intersection of engineering and human communication, and that’s what made it exciting to build.



>Author

Elaina Sara Sabu

ECE + ML Enthusiast

Passionate about signal processing, machine learning, and audio analytics.

:))
