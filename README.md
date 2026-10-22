# Distributed Acoustic TDOA Mesh for Underground Infrastructure Diagnostics

**Author:** Eithan Alexander Valenzuela  
**Architecture:** Embedded DSP (ESP32-S3) · Cascaded SOS Biquad Filtering · Welch-PHAT Cross-Spectral Accumulation · GPS PPS Synchronization

---

## 1. Executive Summary
Municipal water systems lose 15%–30% of treated water to undetected subsurface pipe fractures. Legacy commercial acoustic correlators cost **$3,500–$5,000+ per node**, preventing dense deployment. This project engineers a **~$42 embedded sensor node** and a mathematically rigorous **Band-Limited Welch-PHAT Time-Difference-of-Arrival (TDOA)** solver capable of pinpointing pressurized underground pipe fractures to within **sub-centimeter precision** under severe urban acoustic interference.

---

## 2. Physical & Mathematical Formulation

### 2.1 Wave Kinematics in Pressurized Piping
Turbulent fluid-structure interaction at a fracture site generates broadband elastic stress waves ($150\text{ Hz} - 2,500\text{ Hz}$) that propagate along the pipe wall at velocity $v$ ($1,250\text{ m/s}$ in ductile iron). Given two contact sensors separated by distance $L$, the Time Difference of Arrival ($\Delta t = t_B - t_A$) determines the fracture distance $d_A$ from Sensor A:

$$d_A = \frac{L - v \cdot \Delta t}{2}$$

### 2.2 Numerical Stability via Cascaded Second-Order Sections (SOS)
To prevent floating-point polynomial pole crowding near the complex unit circle ($|z| = 1$) at low cutoff-to-Nyquist ratios, incoming piezoelectric telemetry is conditioned using a 4th-order Butterworth bandpass filter factored into $K = 4$ cascaded biquad sections:

$$H(z) = \prod_{k=1}^{K} \frac{b_{0k} + b_{1k}z^{-1} + b_{2k}z^{-2}}{1 + a_{1k}z^{-1} + a_{2k}z^{-2}}$$

### 2.3 Embedded Welch-PHAT & Cramér-Rao Optimality
Full-buffer Fast Fourier Transforms require $>25\text{ MB}$ of RAM, exceeding the $512\text{ KB}$ internal SRAM of an ESP32-S3 microcontroller. Furthermore, normalizing the cross-spectrum prior to temporal averaging amplifies phase noise at negative SNR. We partition the $16\text{ kHz}$ stream into $M$ overlapping Hann-windowed frames of $N = 4,096$ samples (**96 KB active SRAM footprint**) and accumulate the **Cross-Power Spectral Density (CSD)** prior to Phase Transform (PHAT) whitening:

$$\hat{G}_{BA}(f) = \frac{1}{M} \sum_{m=1}^{M} X_{B,m}(f) X_{A,m}^*(f), \qquad R_{\text{PHAT}}(\tau) = \mathcal{F}^{-1}\left\{ \mathbb{I}(f \in [f_1, f_2]) \frac{\hat{G}_{BA}(f)}{|\hat{G}_{BA}(f)|} \right\}$$

Sub-sample delay resolution is achieved via parabolic vertex interpolation across the discrete correlation maximum $\tau_k$, approaching the **Knapp–Carter Cramér-Rao Lower Bound (CRLB)**:

$$\sigma_{d_A,\text{CRLB}} = \frac{v}{2} \left( 8\pi^2 T \left(\frac{\text{SNR}^2}{1 + 2\cdot\text{SNR}}\right) \left(\frac{f_2^3 - f_1^3}{3}\right) \right)^{-1/2}$$

---

## 3. Empirical Telemetry & Monte Carlo Benchmarks

### 3.1 Single-Capture Spectral Isolation & Cross-Correlation (`tdoa_master.py`)
![TDOA Diagnostic Telemetry](telemetry_proof.png)

### 3.2 Embedded Welch-PHAT vs. Cramér-Rao Lower Bound (`tdoa_embedded_benchmark.py`)
![CRLB Monte Carlo Benchmark](crlb_benchmark.png)

---

## 4. Hardware Node Architecture (~$42 BOM per Node)

| Subsystem | Component | Unit Cost | Technical Role |
| :--- | :--- | :--- | :--- |
| **MCU & DSP** | ESP32-S3-DevKitC-1 (N8R2) | $8.00 | Dual-core 240 MHz Xtensa LX7 with hardware `float32` SIMD FFT acceleration |
| **Clock Sync** | u-blox NEO-6M / NEO-M8N GPS | $12.00 | Hardware **1-PPS (Pulse-Per-Second)** interrupt locking node timers to $<50\text{ ns}$ |
| **Acoustic Transducer** | 27mm Piezoelectric Contact Disc + OPA1678 Pre-Amp | $10.00 | High-impedance structural vibration pickup ($150\text{ Hz} - 2.5\text{ kHz}$) |
| **Coupling & Enclosure** | N52 Neodymium Cup Magnet + IP68 Cast Aluminum Box | $12.00 | Rigid acoustic coupling to cast-iron fire hydrants and curb valves |

---

## 5. Repository Structure & Execution

```bash
pip install numpy scipy matplotlib
python tdoa_master.py               # Runs 3-panel spectral & GCC-PHAT telemetry proof
python tdoa_embedded_benchmark.py   # Runs 300-trial Monte Carlo CRLB stress test