import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt

# Lock random seed for reproducible engineering benchmarks
np.random.seed(42)

# ==============================================================================
# 1. PHYSICAL CONSTANTS & GROUND TRUTH
# ==============================================================================
FS = 44100              # Sampling rate: 44.1 kHz
L = 100.0               # Pipe length between Sensor A and Sensor B (meters)
V = 1250.0              # Acoustic wave velocity in ductile iron pipe (m/s)
TRUE_LEAK_DIST = 32.50  # True leak location from Sensor A (meters)
DURATION = 4.0          # Capture window (seconds)

# Acoustic passband where pressurized pipe-wall vibration lives (Hz)
LOWCUT = 150.0
HIGHCUT = 2500.0

# Exact physical propagation times (seconds)
time_to_A = TRUE_LEAK_DIST / V
time_to_B = (L - TRUE_LEAK_DIST) / V
true_delta_t = time_to_B - time_to_A  # +28.0000 ms (1234.80 samples)

# Maximum physically possible delay is end-to-end pipe travel time (+/- 80 ms)
max_physical_lag = L / V

# ==============================================================================
# 2. SECOND-ORDER SECTIONS (SOS) FILTER & CONTINUOUS WAVE SYNTHESIS
# ==============================================================================
N_samples = int(FS * DURATION)
t_axis = np.arange(N_samples) / FS

def apply_bandpass_sos(data, lowcut, highcut, fs, order=4):
    """
    Zero-phase 4th-order Butterworth bandpass filter using Cascaded Second-Order 
    Sections (SOS / Biquads) to prevent floating-point pole instability (NaN).
    """
    nyq = 0.5 * fs
    sos = butter(order, [lowcut / nyq, highcut / nyq], btype='band', output='sos')
    return sosfiltfilt(sos, data)

def exact_fractional_delay(sig, delay_sec, fs):
    """
    Shifts a signal in Fourier space using X(f) * exp(-j*2*pi*f*tau).
    Preserves exact sub-sample continuous physical travel time.
    """
    N = len(sig)
    X = np.fft.rfft(sig)
    freqs = np.fft.rfftfreq(N, d=1.0 / fs)
    phase_shift = np.exp(-1j * 2.0 * np.pi * freqs * delay_sec)
    return np.fft.irfft(X * phase_shift, n=N)

# Generate physical leak vibration within structural pipe frequency band (RMS = 1.0)
raw_turbulence = np.random.normal(0, 1.0, N_samples)
pure_leak = apply_bandpass_sos(raw_turbulence, LOWCUT, HIGHCUT, FS)
pure_leak /= np.std(pure_leak)

# Propagate wave to Sensor A and Sensor B with exact sub-sample phase delay
clean_A = exact_fractional_delay(pure_leak, time_to_A, FS)
clean_B = exact_fractional_delay(pure_leak, time_to_B, FS)

# Synthesize harsh urban interference (Traffic rumble + 60Hz grid hum + thermal noise)
# 1. Heavy low-frequency street traffic rumble (25 Hz - 110 Hz, 8x louder than leak)
traffic_A = apply_bandpass_sos(np.random.normal(0, 1, N_samples), 25.0, 110.0, FS, order=2)
traffic_B = apply_bandpass_sos(np.random.normal(0, 1, N_samples), 25.0, 110.0, FS, order=2)
traffic_A = (traffic_A / np.std(traffic_A)) * 8.0
traffic_B = (traffic_B / np.std(traffic_B)) * 8.0

# 2. Coherent 60 Hz & 180 Hz electrical grid hum (5x louder than leak)
hum_A = 5.0 * np.sin(2 * np.pi * 60 * t_axis) + 2.0 * np.sin(2 * np.pi * 180 * t_axis)
hum_B = 5.0 * np.sin(2 * np.pi * 60 * t_axis) + 2.0 * np.sin(2 * np.pi * 180 * t_axis)

# 3. Independent broadband sensor noise (2.5x louder than leak)
thermal_A = np.random.normal(0, 2.5, N_samples)
thermal_B = np.random.normal(0, 2.5, N_samples)

# Combine into raw contaminated sensor captures
signal_A = clean_A + traffic_A + hum_A + thermal_A
signal_B = clean_B + traffic_B + hum_B + thermal_B

# Self-check to guarantee zero NaN / numerical overflow
assert np.all(np.isfinite(signal_A)) and np.all(np.isfinite(signal_B)), "Numerical overflow in synthesis!"

raw_snr_db = 20 * np.log10(np.std(clean_A) / np.std(signal_A - clean_A))

# ==============================================================================
# 3. BAND-LIMITED GCC-PHAT & SUB-SAMPLE PARABOLIC INTERPOLATION
# ==============================================================================
# Step 1: Time-domain SOS Butterworth pre-filtering
filtered_A = apply_bandpass_sos(signal_A, LOWCUT, HIGHCUT, FS)
filtered_B = apply_bandpass_sos(signal_B, LOWCUT, HIGHCUT, FS)

def gcc_phat_bandlimited(sig_a, sig_b, fs, lowcut, highcut):
    """
    Computes Band-Limited Generalized Cross-Correlation with Phase Transform.
    Strictly zeroes out-of-band frequencies so PHAT cannot invert the filter stopband.
    """
    n = len(sig_a) + len(sig_b) - 1
    N_fft = 1 << (n - 1).bit_length()

    X_A = np.fft.rfft(sig_a, n=N_fft)
    X_B = np.fft.rfft(sig_b, n=N_fft)
    freqs = np.fft.rfftfreq(N_fft, d=1.0 / fs)

    # Cross-Power Spectrum: G_BA(f) = X_B(f) * conj(X_A(f))
    # Peaks at positive lag tau = +(t_B - t_A)
    cross_spectrum = X_B * np.conj(X_A)

    # Strict spectral mask: only whiten frequencies inside the physical passband
    # Also notch out 180 Hz (3rd harmonic of 60 Hz power grid)
    band_mask = (freqs >= lowcut) & (freqs <= highcut) & (np.abs(freqs - 180.0) > 5.0)

    R = np.zeros_like(cross_spectrum, dtype=np.complex128)
    mag = np.abs(cross_spectrum[band_mask])
    R[band_mask] = cross_spectrum[band_mask] / np.maximum(mag, 1e-12)

    # Inverse FFT back to time-lag domain
    cc = np.fft.irfft(R, n=N_fft)
    cc = np.fft.fftshift(cc)

    # Normalize correlation by number of active passband bins
    cc /= np.sum(band_mask)

    center = N_fft // 2
    lags = np.arange(-center, center) / fs
    return lags, cc, center

def parabolic_interpolation(array, peak_idx):
    """Fits a parabola to the 3 peak samples to resolve sub-sample vertex."""
    if peak_idx <= 0 or peak_idx >= len(array) - 1:
        return float(peak_idx)
    y_m1 = array[peak_idx - 1]
    y_0  = array[peak_idx]
    y_p1 = array[peak_idx + 1]
    denom = y_m1 - 2.0 * y_0 + y_p1
    if abs(denom) < 1e-15:
        return float(peak_idx)
    fractional_offset = 0.5 * (y_m1 - y_p1) / denom
    return float(peak_idx) + fractional_offset

lags, cross_correlation, center_idx = gcc_phat_bandlimited(
    filtered_A, filtered_B, FS, LOWCUT, HIGHCUT
)

# Restrict peak search strictly to physically possible pipe travel times [-L/V, +L/V]
valid_idx = np.where(np.abs(lags) <= max_physical_lag)[0]
local_max_idx = np.argmax(cross_correlation[valid_idx])
discrete_peak_idx = valid_idx[0] + local_max_idx

# Sub-sample refinement via parabolic interpolation
refined_peak_idx = parabolic_interpolation(cross_correlation, discrete_peak_idx)
calc_delta_t = (refined_peak_idx - center_idx) / FS

# Peak-to-Noise-Floor Ratio (Statistical confidence metric)
peak_val = cross_correlation[discrete_peak_idx]
noise_floor_std = np.std(cross_correlation[valid_idx])
z_score_confidence = peak_val / noise_floor_std

# ==============================================================================
# 4. KINEMATIC TRIANGULATION & DIAGNOSTIC TELEMETRY
# ==============================================================================
calc_leak_dist = (L - (V * calc_delta_t)) / 2.0
error_m = abs(TRUE_LEAK_DIST - calc_leak_dist)
error_cm = error_m * 100.0
error_mm = error_m * 1000.0

print("==================================================================")
print(" TDOA ACOUSTIC MESH - DIAGNOSTIC TELEMETRY REPORT")
print("==================================================================")
print(f" [PHYSICAL GROUND TRUTH]")
print(f"   Pipe Length (L):            {L:.2f} m (Wave Velocity: {V:.0f} m/s)")
print(f"   True Leak Position (d_A):   {TRUE_LEAK_DIST:.4f} m")
print(f"   True Time Delay (dt):       {true_delta_t * 1000:.4f} ms ({true_delta_t * FS:.2f} samples)")
print("------------------------------------------------------------------")
print(f" [SIGNAL ENVIRONMENT]")
print(f"   Raw Input SNR:              {raw_snr_db:.2f} dB (Noise is ~9x louder than leak)")
print(f"   Active Passband:            {LOWCUT:.0f} Hz - {HIGHCUT:.0f} Hz (SOS Biquad Cascade)")
print(f"   Peak Detection Confidence:  {z_score_confidence:.1f} sigma (above noise floor)")
print("------------------------------------------------------------------")
print(f" [ALGORITHMIC TRIANGULATION]")
print(f"   Estimated Time Delay (dt):  {calc_delta_t * 1000:.4f} ms ({refined_peak_idx - center_idx:.2f} samples)")
print(f"   Triangulated Leak Position: {calc_leak_dist:.4f} m")
print(f"   Absolute Spatial Error:     {error_cm:.3f} cm ({error_mm:.2f} mm)")
print("==================================================================\n")

# ==============================================================================
# 5. 3-PANEL ENGINEERING DIAGNOSTIC DASHBOARD
# ==============================================================================
plt.style.use('dark_background')
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 8.5))

# Panel 1: Time-Domain Waveforms (First 50 ms)
slice_samples = int(0.050 * FS)
ax1.plot(t_axis[:slice_samples] * 1000, signal_A[:slice_samples], color='#ff5555', alpha=0.6, linewidth=0.8, label='Raw Sensor A (Buried in Traffic + 60Hz Hum)')
ax1.plot(t_axis[:slice_samples] * 1000, filtered_A[:slice_samples], color='#00ffcc', linewidth=1.2, label='SOS Bandpass Filtered Vibration (150-2500 Hz)')
ax1.set_title("1. Time-Domain Sensor Telemetry (50 ms Window)", fontsize=11, fontweight='bold')
ax1.set_xlabel("Time (ms)")
ax1.set_ylabel("Amplitude")
ax1.legend(loc='upper right', fontsize=9)
ax1.grid(True, alpha=0.15)

# Panel 2: Frequency-Domain Power Spectrum (0 - 3500 Hz)
freqs_plot = np.fft.rfftfreq(N_samples, d=1.0 / FS)
psd_raw = 20 * np.log10(np.abs(np.fft.rfft(signal_A)) / N_samples + 1e-12)
psd_filt = 20 * np.log10(np.abs(np.fft.rfft(filtered_A)) / N_samples + 1e-12)
f_mask = freqs_plot <= 3500

ax2.plot(freqs_plot[f_mask], psd_raw[f_mask], color='#ff5555', alpha=0.5, linewidth=0.8, label='Raw Spectrum (Traffic <110Hz + 60/180Hz Spikes)')
ax2.plot(freqs_plot[f_mask], psd_filt[f_mask], color='#00ffcc', linewidth=1.0, label='Isolated Leak Passband')
ax2.axvspan(LOWCUT, HIGHCUT, color='#00ffcc', alpha=0.1, label='Active GCC-PHAT Spectral Mask')
ax2.set_title("2. Frequency-Domain Spectral Isolation (FFT)", fontsize=11, fontweight='bold')
ax2.set_xlabel("Frequency (Hz)")
ax2.set_ylabel("Magnitude (dB)")
ax2.set_ylim(-80, 15)
ax2.legend(loc='upper right', fontsize=9)
ax2.grid(True, alpha=0.15)

# Panel 3: GCC-PHAT Cross-Correlation & Triangulation
lag_mask = np.abs(lags) <= max_physical_lag
ax3.plot(lags[lag_mask] * 1000, cross_correlation[lag_mask], color='#00d4ff', linewidth=1.1, label='GCC-PHAT Correlation')
ax3.axvline(x=calc_delta_t * 1000, color='#ff3366', linestyle='--', linewidth=1.5,
            label=f'Locked Peak: {calc_delta_t*1000:.3f} ms | Est: {calc_leak_dist:.3f} m | Error: {error_cm:.2f} cm ({error_mm:.1f} mm)')
ax3.set_title("3. Band-Limited GCC-PHAT Cross-Correlation (Physical Pipe Window: -80 ms to +80 ms)", fontsize=11, fontweight='bold')
ax3.set_xlabel("Time Difference of Arrival - Lag (ms)")
ax3.set_ylabel("Normalized Correlation")
ax3.legend(loc='upper right', fontsize=9)
ax3.grid(True, alpha=0.15)

plt.tight_layout()
plt.savefig('telemetry_proof.png', dpi=300, bbox_inches='tight')
plt.show()
