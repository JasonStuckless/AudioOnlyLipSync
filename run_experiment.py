from __future__ import annotations

import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict

from implementation1 import run_implementation1 as impl1_run
from implementation2 import run_implementation2
from implementation3 import run_implementation3
from analyze_results import generate_comparative_analysis


def _load_json(path: Path) -> Optional[Dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def print_phase(n: int, title: str) -> None:
    print("\n" + "=" * 70)
    print(f"PHASE {n}: {title}")
    print("=" * 70 + "\n")


def run_complete_experiment(output_dir: str = "experiment_results") -> Optional[dict]:
    logging.info("Starting end-to-end experiment")

    print("\n" + "=" * 70)
    print("END-TO-END EXPERIMENT")
    print("=" * 70)
    print("This will run:")
    print("  1. Implementation 2 (Real-Time Without Buffering)")
    print("  2. Implementation 1 analyzes recorded audio from #1")
    print("  3. Implementation 3 (Real-Time With Buffering)")
    print("  4. Implementation 1 analyzes recorded audio from #3")
    print("  5. Comparative analysis is generated\n")

    input("Press ENTER when ready to begin the experiment.")

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ---------------- Phase 1 — Impl 2 ----------------
    print_phase(1, "Implementation 2: Real-Time Without Buffering")
    config2_file = output_path / f"config2_results_{ts}.json"
    audio2_file = output_path / f"impl2_audio_{ts}.wav"
    input("Press ENTER when ready to begin.")
    results2 = None
    audio2_path = None
    try:
        out = run_implementation2(str(config2_file), str(audio2_file))
        if out is not None:
            results2, audio2_path = out
            print("\n✓ Implementation 2 complete!\n")
        else:
            print("\n✗ Implementation 2 failed\n")
    except Exception as e:
        logging.error(f"Error in Implementation 2: {e}", exc_info=True)
        print(f"\n✗ Implementation 2 failed: {e}\n")

    # ---------------- Phase 2 — Impl 1 for Impl 2 ----------------
    if audio2_path:
        print_phase(2, "Implementation 1: Ground Truth for Implementation 2")
        config1_for_impl2_file = output_path / f"config1_for_impl2_{ts}.json"
        print("Running Implementation 1 on recorded audio from Implementation 2.")
        print(f"Input audio: {audio2_path}")
        print(f"Results: {config1_for_impl2_file}\n")
        try:
            impl1_run(audio2_path, str(config1_for_impl2_file))
            print("\n✓ Implementation 1 analysis complete!\n")
        except Exception as e:
            logging.error(f"Error in Implementation 1 (for impl2): {e}", exc_info=True)
            print(f"\n✗ Implementation 1 failed: {e}\n")
    else:
        print("\nSkipping Implementation 1 analysis — no audio from Implementation 2\n")

    print("\nPausing for 3 seconds before next implementation.\n")
    time.sleep(3)

    # ---------------- Phase 3 — Impl 3 ----------------
    print_phase(3, "Implementation 3: Real-Time With Buffering & Jitter Smoothing")
    config3_file = output_path / f"config3_results_{ts}.json"
    audio3_file = output_path / f"impl3_audio_{ts}.wav"
    input("Press ENTER when ready to begin.")
    results3 = None
    audio3_path = None
    try:
        out = run_implementation3(str(config3_file), str(audio3_file))
        if out is not None:
            results3, audio3_path = out
            print("\n✓ Implementation 3 complete!\n")
        else:
            print("\n✗ Implementation 3 failed\n")
    except Exception as e:
        logging.error(f"Error in Implementation 3: {e}", exc_info=True)
        print(f"\n✗ Implementation 3 failed: {e}\n")

    # ---------------- Phase 4 — Impl 1 for Impl 3 ----------------
    if audio3_path:
        print_phase(4, "Implementation 1: Ground Truth for Implementation 3")
        config1_for_impl3_file = output_path / f"config1_for_impl3_{ts}.json"
        print("Running Implementation 1 on recorded audio from Implementation 3.")
        print(f"Input audio: {audio3_path}")
        print(f"Results: {config1_for_impl3_file}\n")
        try:
            impl1_run(audio3_path, str(config1_for_impl3_file))
            print("\n✓ Implementation 1 analysis complete!\n")
        except Exception as e:
            logging.error(f"Error in Implementation 1 (for impl3): {e}", exc_info=True)
            print(f"\n✗ Implementation 1 failed: {e}\n")
    else:
        print("\nSkipping Implementation 1 analysis — no audio from Implementation 3\n")

    # ---------------- Phase 5 — Comparative analysis ----------------
    print_phase(5, "GENERATING COMPARATIVE ANALYSIS")
    analysis_file = output_path / f"analysis_{ts}.txt"
    analysis: Optional[Dict] = None
    try:
        analysis = generate_comparative_analysis(
            str(output_path / f"config1_for_impl2_{ts}.json"),
            str(output_path / f"config2_results_{ts}.json"),
            str(output_path / f"config1_for_impl3_{ts}.json"),
            str(output_path / f"config3_results_{ts}.json"),
            out_text_path=str(analysis_file),
        )
        if analysis:
            print("\n✓ Analysis complete!\n")
            print(analysis['report'])

            # ---- Additional explicit latency summary ----
            cfg2 = _load_json(config2_file)
            cfg3 = _load_json(config3_file)
            ui2 = analysis.get('ui2') or {}
            ui3 = analysis.get('ui3') or {}

            print("\nLATENCY SUMMARY (Distinct Types)\n" + "-" * 70)
            # Impl-2
            model2 = (cfg2 or {}).get('avg_inference_latency_ms')
            print("Implementation 2 (No Buffering)")
            print("  Model inference latency (audio→detection): ",
                  f"{model2:.2f} ms" if isinstance(model2, (int, float)) else "N/A")
            if ui2:
                print("  UI latency (detection→on‑screen viseme):   ",
                      f"mean {ui2['ui_latency_mean_ms']:.2f} ms  "
                      f"std {ui2['ui_latency_std_ms']:.2f}  min {ui2['ui_latency_min_ms']:.2f}  max {ui2['ui_latency_max_ms']:.2f}")
                if isinstance(model2, (int, float)):
                    print("  End‑to‑end (approx mean):                 ",
                          f"{model2 + ui2['ui_latency_mean_ms']:.2f} ms")
            else:
                print("  UI latency (detection→on‑screen viseme):    N/A")

            # Impl-3
            model3 = (cfg3 or {}).get('avg_inference_latency_ms')
            print("\nImplementation 3 (Buffered & Smoothed)")
            print("  Model inference latency (audio→detection): ",
                  f"{model3:.2f} ms" if isinstance(model3, (int, float)) else "N/A")
            if ui3:
                print("  UI latency (detection→on‑screen viseme):   ",
                      f"mean {ui3['ui_latency_mean_ms']:.2f} ms  "
                      f"std {ui3['ui_latency_std_ms']:.2f}  min {ui3['ui_latency_min_ms']:.2f}  max {ui3['ui_latency_max_ms']:.2f}")
                if isinstance(model3, (int, float)):
                    print("  End‑to‑end (approx mean):                 ",
                          f"{model3 + ui3['ui_latency_mean_ms']:.2f} ms")
            else:
                print("  UI latency (detection→on‑screen viseme):    N/A")
            print("-" * 70 + "\n")
        else:
            print("\n✗ Analysis failed (no data)\n")
    except Exception as e:
        logging.error(f"Analysis error: {e}", exc_info=True)
        print(f"\n✗ Analysis failed: {e}\n")

    return {
        'config2_file': str(config2_file),
        'config1_for_impl2_file': str(output_path / f"config1_for_impl2_{ts}.json"),
        'config3_file': str(config3_file),
        'config1_for_impl3_file': str(output_path / f"config1_for_impl3_{ts}.json"),
        'analysis_file': str(analysis_file) if analysis else None,
    }


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    run_complete_experiment()


if __name__ == "__main__":
    main()