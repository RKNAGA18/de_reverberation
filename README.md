# 🎙️ Single-Channel Speech De-reverberation via SGMSE+

### Team 28  Task 2 Submission

This repository contains the implementation and evaluation of a Score-based Generative Model for Speech Enhancement (SGMSE+) specifically tuned for the task of Speech De-reverberation. Our pipeline transforms reverberant, echoey audio into clean, dry speech using a diffusion-based framework.

---

## 🚀 The Winning Configuration `N=50, SNR=0.33`

After comparing multiple inference settings, we identified Run 2 (N=50, SNR=0.33) as the optimal configuration.

Why we chose this

 Intelligibility It crossed the critical 0.90 STOI threshold, making the speech highly understandable.
 Perceptual Quality It achieved a higher PESQ (2.16) compared to the N=100 run.
 Background Suppression DNSMOS BAK scores reached 4.15, indicating near-perfect removal of late reverberation and noise.

---

## 📊 Final Results Summary

 Metric  Run 1 `n=100, snr=0.5`  Run 2 `n=50, snr=0.33` (Winner) 
 ---  ---  --- 
 PESQ (Quality)  2.095  2.164 
 STOI (Intelligibility)  0.876  0.901 
 SI-SDR (Fidelity)  -2.00 dB  -2.75 dB 
 DNSMOS (BAK)  3.95  4.15 

---

## 🛠️ Methodology & Workflow

### 1. Training & Resilience

 Dataset Clean speech from LibriSpeech convolved with Room Impulse Responses (RIRs) from the ARNI dataset to simulate realistic indoor acoustics.
 The Kaggle Restart Challenge Our initial training run (5,000 samples, 100 epochs) was interrupted by a session timeout. We pivoted by utilizing the official `epoch=326` checkpoint while continuing refined training on a larger dataset, tracked via Weights & Biases (WandB).

### 2. Enhancement Pipeline

1. Inference Generated 60 test samples using the SGMSE+ reverse diffusion process.
2. Temporal Alignment Developed a custom `align_audio.py` script using cross-correlation to fix the tiny time-shifts (latency) inherent in diffusion models.
3. Evaluation Automated the calculation of PESQ, STOI, and SI-SDR using our `sgmse_eval` suite.

---

## 📁 Project Structure

```text
.
├── src
│   ├── enhancement.py     # Main inference script
│   ├── train.py           # Training logic
│   ├── calc_metrics.py    # PESQSTOI calculation
│   ├── calc_dnsmos.py     # DNSMOS Neural Network scoring
│   └── align_audio.py     # Latency correction script
├── results               # Comparison of n100 vs n50 (CSVs & Graphs)
├── checkpoints           # Link to epoch=326-step=408750.ckpt
├── requirements.txt       # Environment setup
└── README.md

```

## ⚙️ Installation & Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Run enhancement with winning settings
python srcenhancement.py --ckpt checkpointsepoch=326.ckpt --N 50 --snr 0.33

# Align and calculate metrics
python srcalign_audio.py
python srccalc_metrics.py

```

---

## 📝 Observations

The SGMSE+ framework effectively suppresses late reverberation components. Our analysis indicates that while higher iteration counts (N=100) provide slightly higher mathematical fidelity, the N=50 configuration preserves more natural speech characteristics, leading to higher human-centric scores (PESQDNSMOS).

---

