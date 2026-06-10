# Adapting Vib2Sound to Missing Accelerometer Cues

**Institute of Neuroinformatics (INI) — University of Zurich & ETH Zurich**
**Supervisor / Lab:** R.H.R. Hahnloser Lab  Mai Akahoshi · Yuhang Wang 
---
## Overview
This project extends [Vib2Sound](https://gitlab.switch.ch/hahnloser-songbird/birdpark/vib2sound) — a neural network system for separating overlapping zebra finch vocalisations using body-mounted acceler. signals and multi-channel microphone recordings — to handle scenarios where **accelerometer cues are partially or entirely missing**.

Vib2Sound was introduced in:
> Akahoshi M., Wang Y., Cheng L., Zai A.T., Hahnloser R.H.R. (2025). *Vib2Sound: Separation of Multimodal Sound Sources*. bioRxiv. https://doi.org/10.1101/2025.05.08.652866

The BirdPark recording system that provides the data is described in:
> Rüttimann L., Wang Y., Rychen J., Tomka T., Hörster H., Hahnloser R.H.R. (2022/2025). *Multimodal system for recording individual-level behaviors in songbird groups*. bioRxiv. https://doi.org/10.1101/2022.09.23.509166

---
## Research Question
> **How can the Vib2Sound separation model be made robust to missing or degraded accelerometer signals, and what strategies best preserve separation performance under partial or total cue absence?**
