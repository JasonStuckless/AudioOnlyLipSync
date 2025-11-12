import time
import logging
import json
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
import cv2
import queue
import threading
import soundfile as sf
import sounddevice as sd

from collections import deque

from components import (
    PhonemeRecognizer,
    IPAToARPAbetConverter,
    PhonemeToVisemeMapper,
    VisemeVisualizer,
)

class Implementation2:
    """Real-time recognition without buffering — displays immediately and logs UI latency."""

    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_duration_ms: int = 500,
        silence_threshold: float = 0.005,
    ) -> None:
        self.sample_rate = sample_rate
        self.chunk_duration_ms = chunk_duration_ms
        self.chunk_size = int(sample_rate * chunk_duration_ms / 1000)
        self.silence_threshold = silence_threshold

        # Window name kept as an attribute for Win32 foreground control
        self.window_name = "Implementation 2: No Buffering"

        # Components
        self.recognizer = PhonemeRecognizer()
        self.ipa_converter = IPAToARPAbetConverter()
        self.mapper = PhonemeToVisemeMapper()
        self.visualizer = VisemeVisualizer(window_name=self.window_name)

        # Audio
        self.audio_queue: "queue.Queue[Tuple[np.ndarray, float]]" = queue.Queue()
        self.audio_stream: Optional[sd.InputStream] = None
        self.recorded_audio_chunks: List[np.ndarray] = []
        self.audio_file_path: Optional[str] = None

        # Threads/state
        self.is_running = False
        self.recognition_thread_obj: Optional[threading.Thread] = None

        # Event queue from recognizer→UI: (viseme:str, arpabet:str, recognition_time:float)
        self.viseme_events: deque[Tuple[str, str, float]] = deque()
        self.lock = threading.Lock()

        # Metrics & logs
        self.log_data: List[Dict] = []           # recognition events
        self.display_events: List[Dict] = []      # what actually got painted
        self.silence_frames = 0
        self.render_count = 0
        self.loop_count = 0
        self.last_viseme: Optional[str] = None
        self.last_arpabet: Optional[str] = None

        logging.info("Implementation 2 initialized (immediate display with UI-latency logging)")

    # ---------------- Foreground helpers ----------------
    def _force_foreground_window(self) -> None:
        try:
            # Ensure window exists in OpenCV's registry
            try:
                cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            except Exception:
                pass
            try:
                cv2.setWindowProperty(self.window_name, cv2.WND_PROP_TOPMOST, 1)
            except Exception:
                pass

            # Win32 fallback using ctypes (no pywin32 dependency)
            import platform
            if platform.system().lower() == 'windows':
                import ctypes
                user32 = ctypes.windll.user32
                kernel32 = ctypes.windll.kernel32
                # Types
                HWND_TOPMOST = -1
                HWND_NOTOPMOST = -2
                SWP_NOMOVE = 0x0002
                SWP_NOSIZE = 0x0001
                SWP_SHOWWINDOW = 0x0040
                SetWindowPos = user32.SetWindowPos
                SetWindowPos.argtypes = [ctypes.wintypes.HWND, ctypes.wintypes.HWND,
                                         ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                         ctypes.c_uint]
                SetWindowPos.restype = ctypes.c_bool
                FindWindowW = user32.FindWindowW
                FindWindowW.argtypes = [ctypes.wintypes.LPCWSTR, ctypes.wintypes.LPCWSTR]
                FindWindowW.restype = ctypes.wintypes.HWND
                SetForegroundWindow = user32.SetForegroundWindow
                SetForegroundWindow.argtypes = [ctypes.wintypes.HWND]
                SetForegroundWindow.restype = ctypes.c_bool

                hwnd = FindWindowW(None, self.window_name)
                if hwnd:
                    # Make it top-most briefly and bring to foreground
                    SetWindowPos(hwnd, HWND_TOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_SHOWWINDOW)
                    SetForegroundWindow(hwnd)
                    # Optionally drop back to not-topmost (keep it visible but polite)
                    SetWindowPos(hwnd, HWND_NOTOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_SHOWWINDOW)
        except Exception as e:
            logging.debug(f"Foreground nudge failed (safe to ignore): {e}")

    # ---------------- Audio callback ----------------
    def audio_callback(self, indata, frames, time_info, status):
        if status:
            logging.warning(f"Audio status: {status}")
        if not self.is_running:
            return
        self.audio_queue.put((indata.copy().flatten(), time.time()))
        self.recorded_audio_chunks.append(indata.copy())

    # --------------- Recognition worker ---------------
    def _emit(self, arpabet: str, recognition_time: float):
        # Map to viseme and push to event queue; UI loop will paint and timestamp display_time
        viseme = self.mapper.phoneme_to_viseme(arpabet)
        with self.lock:
            # Dedup exact same pair to avoid visual flicker
            if viseme == self.last_viseme and arpabet == self.last_arpabet:
                return
            self.viseme_events.append((viseme, arpabet, recognition_time))
            self.last_viseme = viseme
            self.last_arpabet = arpabet

    def _handle_phoneme(self, ipa_or_seg: str, rec_t: float, audio_t: float, energy: float) -> None:
        arpabet = self.ipa_converter.convert(ipa_or_seg)
        if not arpabet:
            return
        # Log recognition event (pre-display)
        self.log_data.append({
            'audio_time': audio_t,
            'recognition_time': rec_t,
            'ipa_phoneme': ipa_or_seg,
            'arpabet_phoneme': arpabet,
            'inference_latency': (rec_t - audio_t) * 1000.0,
            'energy': float(energy),
            'silence_detected': False,
        })
        self._emit(arpabet, rec_t)

    def recognition_thread(self) -> None:
        logging.info("Recognition thread started")
        while self.is_running:
            try:
                audio_chunk, audio_timestamp = self.audio_queue.get(timeout=1.0)
                energy = float(np.abs(audio_chunk).mean())
                if energy < self.silence_threshold:
                    self.silence_frames += 1
                    # Optional: log silence (no display event)
                    self.log_data.append({
                        'audio_time': audio_timestamp,
                        'recognition_time': time.time(),
                        'ipa_phoneme': 'SIL',
                        'arpabet_phoneme': 'SIL',
                        'inference_latency': (time.time() - audio_timestamp) * 1000.0,
                        'energy': float(energy),
                        'silence_detected': True,
                    })
                    continue
                # Recognize IPA
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
                                self._handle_phoneme(s, rec_t, audio_timestamp, energy)
                            continue
                    self._handle_phoneme(ipa, rec_t, audio_timestamp, energy)
            except queue.Empty:
                if not self.is_running:
                    break
                continue
            except Exception as e:
                logging.error(f"Recognition error: {e}", exc_info=True)
        logging.info("Recognition thread completed")

    # ---------------- Lifecycle ----------------
    def start(self):
        logging.info("=" * 60)
        logging.info("IMPLEMENTATION 2: No Buffering (display + UI latency logging)")
        logging.info("=" * 60)
        print("\n" + "=" * 60)
        print("🎤 READY TO START (Implementation 2)")
        print("Press 'Q' in the window to stop")
        print("=" * 60 + "\n")

        self.is_running = True
        self.recognition_thread_obj = threading.Thread(target=self.recognition_thread, daemon=True)
        self.recognition_thread_obj.start()

        # Initial frame and warm-up to create & show the window
        self.visualizer.update_display("silence", "")
        for _ in range(15):
            self.visualizer.render()
            cv2.waitKey(1)
        # Nudge to foreground/top-most (mirrors Impl-3 behavior more aggressively)
        self._force_foreground_window()

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

        # Nudge again once the stream is live (some desktops steal focus when devices activate)
        self._force_foreground_window()

        # UI loop — drain viseme events immediately; each event triggers a paint
        try:
            while self.is_running:
                self.loop_count += 1
                painted = False
                while True:
                    with self.lock:
                        if not self.viseme_events:
                            break
                        viseme, arpabet, rec_t = self.viseme_events.popleft()
                    # Paint this event now
                    self.visualizer.update_display(viseme, arpabet)
                    disp_t = time.time()
                    if self.visualizer.render():
                        painted = True
                        self.render_count += 1
                    # Log actual display event and UI latency
                    self.display_events.append({
                        'arpabet_phoneme': arpabet,
                        'display_time': disp_t,
                        'ui_latency_ms': (disp_t - rec_t) * 1000.0,
                    })
                    # Let the GUI breathe
                    cv2.waitKey(1)
                if not painted:
                    # keep window alive/responsive
                    self.visualizer.render()
                    cv2.waitKey(1)
                # Quit?
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
        except KeyboardInterrupt:
            logging.info("Interrupted by user")
        finally:
            self.stop()

    def save_audio_recording(self, output_file: str = "impl2_audio.wav") -> Optional[str]:
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
        logging.info("Stopping Implementation 2.")
        self.is_running = False
        try:
            if self.audio_stream is not None:
                self.audio_stream.stop()
                self.audio_stream.close()
        except Exception:
            pass
        self.save_audio_recording(self.audio_file_path or "impl2_audio.wav")
        self.visualizer.close()
        logging.info("Implementation 2 stopped")

    def get_results(self) -> Optional[Dict]:
        if not self.log_data and not self.display_events:
            return None
        lat = [d['inference_latency'] for d in self.log_data if not d.get('silence_detected')]
        ui_lat = [d['ui_latency_ms'] for d in self.display_events if 'ui_latency_ms' in d]
        results = {
            'avg_inference_latency_ms': float(np.mean(lat)) if lat else 0.0,
            'min_inference_latency_ms': float(np.min(lat)) if lat else 0.0,
            'max_inference_latency_ms': float(np.max(lat)) if lat else 0.0,
            'phoneme_events': int(sum(1 for d in self.log_data if not d.get('silence_detected'))),
            'silence_frames': int(self.silence_frames),
            'render_count': int(self.render_count),
            'loop_count': int(self.loop_count),
            # provide explicit sequence used by analyzer
            'phoneme_sequence': list(self.display_events),
            # summary of UI latency
            'ui_latency_mean_ms': float(np.mean(ui_lat)) if ui_lat else 0.0,
            'ui_latency_std_ms': float(np.std(ui_lat)) if ui_lat else 0.0,
            'ui_latency_min_ms': float(np.min(ui_lat)) if ui_lat else 0.0,
            'ui_latency_max_ms': float(np.max(ui_lat)) if ui_lat else 0.0,
        }
        return results


def run_implementation2(results_path: str, audio_path: str) -> Optional[Tuple[dict, str]]:
    impl = Implementation2()
    impl.audio_file_path = audio_path
    impl.start()  # blocks until 'q'
    results = impl.get_results()
    if results is None:
        return None
    try:
        rp = Path(results_path)
        rp.parent.mkdir(parents=True, exist_ok=True)
        with open(rp, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
    except Exception as e:
        logging.error(f"Failed to save impl2 results: {e}")
    audio_file = impl.save_audio_recording(audio_path) or audio_path
    return results, audio_file