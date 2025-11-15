# End-to-End Real-Time Phoneme-to-Viseme System  
### Offline Ground Truth • Real-Time Capture • Buffered/Jitter-Smoothed Pipeline • Comparative Evaluation

This repository contains a complete experimental framework for real-time phoneme recognition and viseme animation, including:

1. Implementation 1 — Offline Ground Truth (high accuracy, not real-time)  
2. Implementation 2 — Real-Time, No Buffering (immediate display, logs UI latency)  
3. Implementation 3 — Real-Time, Buffered & Jitter-Smoothed (stable, smoothed output)

An end-to-end experiment runner executes all stages, records audio, builds ground truth, and generates a comparative analysis.

------------------------------------------------------------
FEATURES
------------------------------------------------------------

• Real-time CTC phoneme recognition (wav2vec2)  
• IPA → ARPAbet conversion with segmentation  
• 15-category viseme mapping  
• Live sprite rendering using OpenCV  
• Offline overlapping-window ground truth generation  
• Real-time UI-latency measurement  
• Buffer-based timing stabilization (Impl-3)  
• Automated comparative accuracy & latency report

------------------------------------------------------------
PROJECT STRUCTURE
------------------------------------------------------------

    .
    ├── components.py                # Core converter, recognizer, mapper, visualizer
    ├── implementation1.py           # Offline ground truth system
    ├── implementation2.py           # Real-time system (no buffering)
    ├── implementation3.py           # Real-time system (buffered & smoothed)
    ├── analyze_results.py           # Accuracy/timing/UI-latency analysis
    ├── run_experiment.py            # Full pipeline controller
    ├── viseme_sprites/              # Viseme PNG/JPG sprites or placeholders
    └── README.md

------------------------------------------------------------
QUICK START
------------------------------------------------------------

1. Install dependencies

    pip install -r requirements.txt

2. Run the full end-to-end experiment

    python run_experiment.py

Outputs appear under:

    experiment_results/

------------------------------------------------------------
RUNNING INDIVIDUAL IMPLEMENTATIONS
------------------------------------------------------------

Implementation 1 (offline baseline):

    python implementation1.py

Implementation 2 (real-time, no buffering):

    python implementation2.py

Implementation 3 (real-time, buffered):

    python implementation3.py

------------------------------------------------------------
OUTPUT FILES
------------------------------------------------------------

Each run produces:

• JSON logs (phonemes, timings, visemes)  
• Recorded audio WAV  
• Comparative analysis report

Example filenames:

    config2_results_YYYYMMDD_HHMMSS.json
    impl2_audio_YYYYMMDD_HHMMSS.wav
    analysis_YYYYMMDD_HHMMSS.txt

------------------------------------------------------------
VISEME SPRITES
------------------------------------------------------------

Place viseme images here:

    viseme_sprites/
        silence.png
        bilabial_stop.png
        labiodental.png
        ...

Missing sprites are auto-generated as placeholders though this should not need to occur since all are provided in the corresponding folder.

------------------------------------------------------------
CONFIGURATION
------------------------------------------------------------

Implementation 1:
• Overlap percentage  
• Chunk size  
• Silence threshold  

Implementation 2:
• Chunk size  
• Immediate display behavior  
• UI latency logging  

Implementation 3:
• Target latency (ms)  
• Target inter-viseme interval (ms)  
• Jitter smoothing settings  
