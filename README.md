# Dog-Harness-OS

A backend algorithm system for canine health and behavior monitoring.

## Architecture
This system follows an Edge-to-Cloud structure. The physical collar collects raw sensor data (IMU, Mic, HR/HRV), and this repository acts as the cloud brain. It processes the raw data to output physiological and psychological health status.

## Core Modules
* **Transient Crisis Engine (M1, M3):** Real-time detection of acute stress, startles, and physical hits. 
* **Metabolic Continuum (M4.1):** 48-hour continuous cortisol (stress hormone) tracking.
* **Cardio & Temperament Radar (M2, M4-C):** Evaluates emotional recovery speed and cardiopulmonary aging from daily events.
* **Chronic Disease Predictor (M4-B, M4.5):** Long-term data analysis to predict the onset of chronic pain (e.g., osteoarthritis).

## Quick Start

1. Install required tools:
   pip install -r requirements.txt

2. Run the main program to test the system:
   python main.py

## License
Strictly Non-Commercial. See LICENSE file for details.