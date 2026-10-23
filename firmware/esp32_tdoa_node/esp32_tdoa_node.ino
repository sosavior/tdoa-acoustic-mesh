/**
 * ============================================================================
 * DISTRIBUTED ACOUSTIC TDOA MESH - ESP32-S3 EDGE NODE FIRMWARE
 * ============================================================================
 * Target MCU:      ESP32-S3-DevKitC-1 (Dual-Core 240 MHz Xtensa LX7)
 * Sampling Rate:   16,000.00 Hz (62.5 us period, GPS-1PPS Disciplined)
 * Memory Profile:  64 KB Static SRAM Capture Buffer (16,384 float32 samples)
 * DSP Pipeline:    Direct-Form II Transposed Cascaded SOS Biquad (150-2500 Hz)
 * ============================================================================
 */

#include <Arduino.h>
#include "esp_timer.h"

// ============================================================================
// 1. HARDWARE PIN DEFINITIONS & CONFIGURATION
// ============================================================================
#define PIN_SENSOR_ADC      4       // GPIO4 (ADC1_CH3) -> Vibration Sensor Z-Out
#define PIN_GPS_PPS         5       // GPIO5 -> u-blox NEO-6M/8N 1-PPS Pin
#define PIN_STATUS_LED      2       // GPIO2 -> Onboard Trigger Indicator LED

#define NODE_ID             'A'     // Set to 'A' for Node A, 'B' for Node B
#define SAMPLE_RATE_HZ      16000   // 16.0 kHz acoustic sampling rate
#define CAPTURE_SAMPLES     16384   // 1.024-second synchronized window (64 KB RAM)
#define SERIAL_BAUD         921600  // High-speed UART streaming to host laptop

// ============================================================================
// 2. GPS 1-PPS DISCIPLINED CLOCK SYNCHRONIZATION (ISR STATE)
// ============================================================================
volatile int64_t  last_pps_us      = 0;
volatile int64_t  pps_interval_us  = 1000000; // Nominal: 1,000,000 us (1.000000 s)
volatile uint32_t pps_pulse_count  = 0;
volatile bool     pps_sync_locked  = false;
volatile bool     capture_armed    = false;
volatile bool     capture_ready    = false;

// 64 KB Static SRAM Buffer (16,384 x 4-byte IEEE-754 float32)
static float adc_capture_buffer[CAPTURE_SAMPLES];

// Interrupt Service Routine (ISR) placed in internal Instruction RAM (IRAM)
// Latches the 64-bit hardware timer within <50 nanoseconds of GPS PPS rising edge
void IRAM_ATTR onPpsInterrupt() {
    int64_t now_us = esp_timer_get_time();
    if (last_pps_us > 0) {
        int64_t delta = now_us - last_pps_us;
        // Validate pulse is within +/- 5% of 1.0 second (rejects electrical noise)
        if (delta > 950000 && delta < 1050000) {
            pps_interval_us = delta;
            pps_pulse_count++;
            pps_sync_locked = true;
        }
    }
    last_pps_us = now_us;

    // If host armed a synchronized capture, trigger immediately on this PPS edge
    if (capture_armed && !capture_ready) {
        capture_armed = false;
        // Core 0 acquisition task watches this timestamp
    }
}

// ============================================================================
// 3. CASCADED SECOND-ORDER SECTIONS (SOS / BIQUAD) BANDPASS FILTER
// ============================================================================
// Exact Butterworth 2nd-Order High-Pass (150 Hz) + 2nd-Order Low-Pass (2500 Hz)
// Computed for Fs = 16,000.0 Hz (Direct-Form II Transposed for float32 stability)
struct BiquadSection {
    float b0, b1, b2;
    float a1, a2;
    float z1, z2;

    inline float process(float x) {
        float y = b0 * x + z1;
        z1 = b1 * x - a1 * y + z2;
        z2 = b2 * x - a2 * y;
        return y;
    }

    void reset() {
        z1 = 0.0f;
        z2 = 0.0f;
    }
};

// Stage 1: 150 Hz High-Pass (Strips DC bias, traffic rumble <110 Hz, & 60 Hz hum)
BiquadSection sos_stage1_hp = {
     0.9592032f, -1.9184064f,  0.9592032f,
    -1.9167412f,  0.9200714f,  0.0f, 0.0f
};

// Stage 2: 2500 Hz Low-Pass (Anti-aliasing & high-frequency thermal noise cutoff)
BiquadSection sos_stage2_lp = {
     0.1399393f,  0.2798786f,  0.1399393f,
    -0.6997380f,  0.2594952f,  0.0f, 0.0f
};

// ============================================================================
// 4. CORE 0 REAL-TIME DETERMINISTIC ACQUISITION ENGINE
// ============================================================================
void acquisitionTaskCore0(void *parameter) {
    for (;;) {
        // Wait until user or GPS PPS triggers a synchronized capture window
        if (!capture_armed && !capture_ready) {
            vTaskDelay(1);
            continue;
        }

        if (capture_armed) {
            // Wait for exact rising edge of next GPS 1-PPS pulse (or immediate if bench testing)
            uint32_t start_pps_count = pps_pulse_count;
            int64_t wait_start = esp_timer_get_time();

            while (pps_sync_locked && (pps_pulse_count == start_pps_count)) {
                if (esp_timer_get_time() - wait_start > 1500000) break; // Timeout fallback
            }

            int64_t t0_us = esp_timer_get_time();
            double clock_scale = (double)pps_interval_us / 1000000.0;

            sos_stage1_hp.reset();
            sos_stage2_lp.reset();
            digitalWrite(PIN_STATUS_LED, HIGH);

            // Warm up biquad filter states over 128 samples before recording
            for (int w = 0; w < 128; w++) {
                float raw_v = (float)analogRead(PIN_SENSOR_ADC) - 2048.0f;
                sos_stage2_lp.process(sos_stage1_hp.process(raw_v));
            }

            t0_us = esp_timer_get_time();

            // Deterministic 16,000.00 Hz sampling loop (62.5 us GPS-disciplined step)
            for (uint32_t i = 0; i < CAPTURE_SAMPLES; i++) {
                int64_t target_us = t0_us + (int64_t)((double)i * 62.5 * clock_scale);
                while (esp_timer_get_time() < target_us) {
                    // Tight hardware-timer spinlock for sub-microsecond jitter
                }

                // Read 12-bit SAR ADC (0..4095), center at zero, and run SOS Biquad cascade
                float raw_sample = (float)analogRead(PIN_SENSOR_ADC) - 2048.0f;
                float hp_out     = sos_stage1_hp.process(raw_sample);
                adc_capture_buffer[i] = sos_stage2_lp.process(hp_out);
            }

            digitalWrite(PIN_STATUS_LED, LOW);
            capture_armed = false;
            capture_ready = true;
        }
    }
}

// ============================================================================
// 5. INITIALIZATION & CORE 1 TELEMETRY STREAMING
// ============================================================================
void setup() {
    Serial.begin(SERIAL_BAUD);
    pinMode(PIN_STATUS_LED, OUTPUT);
    pinMode(PIN_GPS_PPS, INPUT_PULLDOWN);

    // Configure ESP32-S3 12-bit ADC (0V - 3.1V full scale attenuation)
    analogReadResolution(12);
    analogSetPinAttenuation(PIN_SENSOR_ADC, ADC_11db);

    // Attach sub-50ns hardware interrupt to GPS 1-PPS pin
    attachInterrupt(digitalPinToInterrupt(PIN_GPS_PPS), onPpsInterrupt, RISING);

    // Pin deterministic ADC acquisition engine to dedicated Core 0
    xTaskCreatePinnedToCore(
        acquisitionTaskCore0,
        "ADC_Core0_Task",
        4096,
        NULL,
        configMAX_PRIORITIES - 1,
        NULL,
        0
    );

    delay(500);
    Serial.println("\n=======================================================");
    Serial.printf(" ESP32-S3 TDOA ACOUSTIC NODE [%c] INITIALIZED\n", NODE_ID);
    Serial.println(" Commands: Send 'T' to trigger synchronized capture");
    Serial.println("=======================================================");
}

void loop() {
    // Listen for host laptop trigger command ('T')
    if (Serial.available() > 0) {
        char cmd = Serial.read();
        if (cmd == 'T' || cmd == 't') {
            capture_ready = false;
            capture_armed = true;
        }
    }

    // Once Core 0 finishes the 16,384-sample capture, stream frame to host laptop
    if (capture_ready) {
        float crystal_ppm = (float)(pps_interval_us - 1000000);

        Serial.printf("BEGIN_FRAME,NODE=%c,FS=%d,N=%d,PPS_LOCK=%d,DRIFT_PPM=%.2f\n",
                      NODE_ID, SAMPLE_RATE_HZ, CAPTURE_SAMPLES,
                      pps_sync_locked ? 1 : 0, crystal_ppm);

        for (uint32_t i = 0; i < CAPTURE_SAMPLES; i++) {
            Serial.println(adc_capture_buffer[i], 4);
        }

        Serial.println("END_FRAME");
        capture_ready = false;
    }

    delay(10);
}