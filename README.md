<div align="center">

# ⌨️ Typing Biometrics

![Typing SVG](https://readme-typing-svg.demolab.com?font=Inter&weight=600&size=35&pause=1000&color=3B82F6&center=true&vCenter=true&width=800&lines=Keystroke+Dynamics;Behavioral+Biometrics;Continuous+Authentication)

[![Python](https://img.shields.io/badge/Python-3.12-3B82F6?style=for-the-badge&logo=python&logoColor=white)](#)
[![Flask](https://img.shields.io/badge/Flask-Web_Framework-black?style=for-the-badge&logo=flask)](#)
[![Security](https://img.shields.io/badge/Security-Zero_Trust-00C853?style=for-the-badge&logo=security)](#)

</div>

---

## 🌟 Overview

Welcome to **Typing Biometrics**, an advanced authentication system that verifies users not by *what they type*, but by **how they type**. By analyzing keystroke dynamics—including dwell times, flight times between keys, and statistical digraph/trigraph patterns—this project creates a unique "behavioral fingerprint" for each user.

No neural networks. No opaque models. Just high-dimensional, interpretable statistical fingerprinting.

<br>

<div align="center">
  <img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" width="100%">
</div>

## ✨ Core Features

<details>
  <summary><b>🔍 High-Dimensional Fingerprinting</b></summary>
  <br>
  Generates detailed per-key and per-digraph statistical profiles (400-600 dimensions) to capture individual typing cadence accurately.
</details>

<details>
  <summary><b>⚡ Real-Time Verification</b></summary>
  <br>
  Processes raw <code>keydown</code> and <code>keyup</code> events directly from the browser with millisecond precision, computing overlap, flight, and dwell durations on the fly.
</details>

<details>
  <summary><b>🛡️ Adaptive Distance Metrics</b></summary>
  <br>
  Calibrates naturally to user variance, using intersection-based relative difference scoring to gracefully handle sparse arrays and differing vocabularies.
</details>

<br>

## 🚀 Architecture Details

Instead of relying on basic averages (like WPM) or black-box neural models, this system extracts an exhaustive set of statistical markers from every keystroke sequence:

1. **Event Parsing:** Translates raw browser events into clean `(key, dwell, flight)` tuples.
2. **Feature Extraction:** Computes mean, std dev, min, and max for every key, digraph, and trigraph present in the input.
3. **Structural Profiling:** Measures shift-overlap patterns, spacebar pausing, and fatigue vectors (sentence-third comparisons).
4. **Scoring:** Compares the intersection of active dimensions between profiles using a normalized relative difference algorithm.

> *"Two people typing 'Hey there' might have the exact same speed, but their flight time for the 'th' transition and dwell on 'e' will uniquely identify them."*

## 🛠️ Tech Stack

- **Frontend:** HTML5, CSS3, Vanilla JS (Event capturing via `performance.now()`)
- **Backend:** Python, Flask
- **Data/Math:** Numpy, Pandas, SQLite

---

<div align="center">
  <i>Built for seamless, continuous behavioral authentication.</i>
</div>
