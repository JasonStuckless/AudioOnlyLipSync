import time
import logging
import json
from pathlib import Path
from collections import deque
from typing import Optional, Tuple, List, Dict

import numpy as np
import cv2
import threading
import queue
import soundfile as sf
import sounddevice as sd

from components import (
    PhonemeRecognizer,
    IPAToARPAbetConverter,
    PhonemeToVisemeMapper,
    VisemeVisualizer,
)

class LatencyBuffer:
    def __init__(self, target_latency_ms: int = 100, target_interval_ms: int = 100):
        self.target_latency = target_latency_ms / 1000.0
        self.target_interval = target_interval_ms / 1000.0
        self.buffer: deque[Tuple[str, float]] = deque()  # (arpabet, recognition_time)
        self.lock = threading.Lock()
        self.last_display_time: Optional[float] = None
        self.release_times: List[float] = []
        self.held_count = 0

    def add(self, arpabet: str, recognition_time: float) -> None:
        with self.lock:
            self.buffer.append((arpabet, recognition_time))

    def get_ready(self, now: float) -> Optional[Tuple[str, float]]:
        with self.lock:
            if not self.buffer:
                return None
            arpabet, rec_t = self.buffer[0]
            if now - rec_t < self.target_latency:
                return None
            if self.last_display_time is not None:
                dt = now - self.last_display_time
                if dt < self.target_interval:
                    self.held_count += 1
                    return None
            # release
            self.buffer.popleft()
            if self.last_display_time is not None:
                self.release_times.append(now - self.last_display_time)
            self.last_display_time = now
            return arpabet, rec_t

    def jitter_stats(self) -> Dict[str, float]:
        if not self.release_times:
            return {
                'mean_interval': 0.0,
                'std_interval': 0.0,
                'min_interval': 0.0,
                'max_interval': 0.0,
                'held_count': int(self.held_count),
                'jitter_coefficient': 0.0,
            }
        arr = [t * 1000.0 for t in self.release_times]
        mean = float(np.mean(arr))
        std = float(np.std(arr))
        return {
            'mean_interval': mean,
            'std_interval': std,
            'min_interval': float(np.min(arr)),
            'max_interval': float(np.max(arr)),
            'held_count': int(self.held_count),
            'jitter_coefficient': (std / mean * 100.0) if mean > 0 else 0.0,
        }


class Implementation3:
    def __init__(
        self,
        target_latency_ms: int = 100,
        target_interval_ms: int = 100,
        sample_rate: int = 16000,
        chunk_duration_ms: int = 500,
        silence_threshold: float = 0.005,
    ) -> None:
        self.sample_rate = sample_rate
        self.chunk_duration_ms = chunk_duration_ms
        self.chunk_size = int(sample_rate * chunk_duration_ms / 1000)
        self.silence_threshold = silence_threshold

        self.recognizer = PhonemeRecognizer()
        self.ipa_converter = IPAToARPAbetConverter()
        self.mapper = PhonemeToVisemeMapper()
        self.visualizer = VisemeVisualizer(window_name="Implementation 3: Buffered & Smoothed")

        self.audio_queue: "queue.Queue[Tuple[np.ndarray, float]]" = queue.Queue()
        self.audio_stream: Optional[sd.InputStream] = None
        self.recorded_audio_chunks: List[np.ndarray] = []
        self.audio_file_path: Optional[str] = None

        self.latency_buffer = LatencyBuffer(target_latency_ms, target_interval_ms)
        self.target_latency_ms = target_latency_ms
        self.target_interval_ms = target_interval_ms

        self.is_running = False
        self.recognition_thread_obj: Optional[threading.Thread] = None

        self.log_data: List[Dict] = []        # recognition events
        self.display_events: List[Dict] = []   # on-screen events
        self.last_viseme: Optional[str] = None
        self.last_arpabet: Optional[str] = None
        self.silence_frames = 0
        self.render_count = 0
        self.loop_count = 0

        logging.info("Implementation 3 initialized (records detection→display latency)")

    # ---- audio ----
    def audio_callback(self, indata, frames, time_info, status):
        if status:
            logging.warning(f"Audio status: {status}")
        if not self.is_running:
            return
        self.audio_queue.put((indata.copy().flatten(), time.time()))
        self.recorded_audio_chunks.append(indata.copy())

    # ---- recognition ----
    def _enqueue(self, arpabet: str, rec_t: float, audio_t: float, energy: float) -> None:
        self.latency_buffer.add(arpabet, rec_t)
        self.log_data.append({
            'audio_time': audio_t,
            'recognition_time': rec_t,
            'ipa_phoneme': None,  # optional; Impl-3 focuses on ARPAbet path
            'arpabet_phoneme': arpabet,
            'inference_latency': (rec_t - audio_t) * 1000.0,
            'energy': float(energy),
            'silence_detected': False,
        })

    def recognition_thread(self) -> None:
        logging.info("Recognition thread started")
        while self.is_running:
            try:
                audio_chunk, audio_t = self.audio_queue.get(timeout=1.0)
                energy = float(np.abs(audio_chunk).mean())
                if energy < self.silence_threshold:
                    self.silence_frames += 1
                    self.log_data.append({
                        'audio_time': audio_t,
                        'recognition_time': time.time(),
                        'ipa_phoneme': 'SIL',
                        'arpabet_phoneme': 'SIL',
                        'inference_latency': (time.time() - audio_t) * 1000.0,
                        'energy': float(energy),
                        'silence_detected': True,
                    })
                    continue
                try:
                    ipa_list = self.recognizer.recognize(audio_chunk)
                except TypeError:
                    ipa_list = self.recognizer.recognize(audio_chunk, self.sample_rate)
                rec_t = time.time()
                for ipa in ipa_list:
                    if len(ipa) > 1:
                        segs = self.ipa_converter.segment_ipa_string(ipa)
                        if len(segs) > 1:
                            for s in segs:
                                arp = self.ipa_converter.convert(s)
                                if arp:
                                    self._enqueue(arp, rec_t, audio_t, energy)
                            continue
                    arp = self.ipa_converter.convert(ipa)
                    if arp:
                        self._enqueue(arp, rec_t, audio_t, energy)
            except queue.Empty:
                if not self.is_running:
                    break
                continue
            except Exception as e:
                logging.error(f"Recognition error: {e}", exc_info=True)
        logging.info("Recognition thread completed")

    # ---- lifecycle ----
    def start(self) -> None:
        logging.info("=" * 60)
        logging.info("IMPLEMENTATION 3: Buffered & Smoothed (with UI-latency logging)")
        logging.info("=" * 60)
        print("\n" + "=" * 60)
        print("🎤 READY TO START (Implementation 3)")
        print("Press 'Q' in the window to stop")
        print("=" * 60 + "\n")

        self.is_running = True
        self.recognition_thread_obj = threading.Thread(target=self.recognition_thread, daemon=True)
        self.recognition_thread_obj.start()

        # Initial frame; warm-up to foreground
        self.visualizer.update_display("silence", "")
        for _ in range(15):
            self.visualizer.render()
            cv2.waitKey(1)

        # Start audio
        try:
            self.audio_stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype='float32',
                blocksize=self.chunk_size,
                callback=self.audio_callback,
            )
            self.audio_stream.start()
            logging.info("Audio stream started")
        except Exception as e:
            logging.error(f"Failed to start audio stream: {e}", exc_info=True)
            self.is_running = False
            return

        # UI loop — paced release via latency buffer
        try:
            while self.is_running:
                self.loop_count += 1
                now = time.time()
                ready = self.latency_buffer.get_ready(now)
                if ready:
                    arpabet, rec_t = ready
                    viseme = self.mapper.phoneme_to_viseme(arpabet)
                    # Dedup last state
                    if viseme != self.last_viseme or arpabet != self.last_arpabet:
                        self.visualizer.update_display(viseme, arpabet)
                        disp_t = time.time()
                        if self.visualizer.render():
                            self.render_count += 1
                        # record actual display + ui latency
                        self.display_events.append({
                            'arpabet_phoneme': arpabet,
                            'display_time': disp_t,
                            'ui_latency_ms': (disp_t - rec_t) * 1000.0,
                        })
                        self.last_viseme = viseme
                        self.last_arpabet = arpabet
                else:
                    self.visualizer.render()
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
        except KeyboardInterrupt:
            logging.info("Interrupted by user")
        finally:
            self.stop()

    def save_audio_recording(self, output_file: str = "impl3_audio.wav") -> Optional[str]:
        if not self.recorded_audio_chunks:
            logging.warning("No audio recorded to save")
            return None
        try:
            audio_data = np.concatenate(self.recorded_audio_chunks, axis=0)
            if audio_data.ndim > 1:
                audio_data = audio_data.flatten()
            sf.write(output_file, audio_data, self.sample_rate)
            self.audio_file_path = output_file
            duration = len(audio_data) / self.sample_rate
            logging.info(f"Saved {duration:.2f}s of audio to {output_file}")
            return output_file
        except Exception as e:
            logging.error(f"Failed to save audio recording: {e}")
            return None

    def stop(self) -> None:
        logging.info("Stopping Implementation 3.")
        self.is_running = False
        try:
            if self.audio_stream is not None:
                self.audio_stream.stop()
                self.audio_stream.close()
        except Exception:
            pass
        self.save_audio_recording(self.audio_file_path or "impl3_audio.wav")
        self.visualizer.close()
        logging.info("Implementation 3 stopped")

    def get_results(self) -> Optional[Dict]:
        if not self.log_data and not self.display_events:
            return None
        lat = [d['inference_latency'] for d in self.log_data if not d.get('silence_detected')]
        ui_lat = [d['ui_latency_ms'] for d in self.display_events if 'ui_latency_ms' in d]
        jitter = self.latency_buffer.jitter_stats()
        return {
            'avg_inference_latency_ms': float(np.mean(lat)) if lat else 0.0,
            'min_inference_latency_ms': float(np.min(lat)) if lat else 0.0,
            'max_inference_latency_ms': float(np.max(lat)) if lat else 0.0,
            'phoneme_events': int(sum(1 for d in self.log_data if not d.get('silence_detected'))),
            'silence_frames': int(sum(1 for d in self.log_data if d.get('silence_detected'))),
            'render_count': int(self.render_count),
            'loop_count': int(self.loop_count),
            'target_latency_ms': int(self.target_latency_ms),
            'target_interval_ms': int(self.target_interval_ms),
            'min_interval_ms': float(jitter['min_interval']),
            'max_interval_ms': float(jitter['max_interval']),
            'jitter_coefficient': float(jitter['jitter_coefficient']),
            'phoneme_sequence': list(self.display_events),
            'ui_latency_mean_ms': float(np.mean(ui_lat)) if ui_lat else 0.0,
            'ui_latency_std_ms': float(np.std(ui_lat)) if ui_lat else 0.0,
            'ui_latency_min_ms': float(np.min(ui_lat)) if ui_lat else 0.0,
            'ui_latency_max_ms': float(np.max(ui_lat)) if ui_lat else 0.0,
        }


def run_implementation3(results_path: str, audio_path: str) -> Optional[Tuple[dict, str]]:
    impl = Implementation3()
    impl.audio_file_path = audio_path
    impl.start()
    results = impl.get_results()
    if results is None:
        return None
    try:
        rp = Path(results_path)
        rp.parent.mkdir(parents=True, exist_ok=True)
        with open(rp, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
    except Exception as e:
        logging.error(f"Failed to save impl3 results: {e}")
    audio_file = impl.save_audio_recording(audio_path) or audio_path
    return results, audio_file