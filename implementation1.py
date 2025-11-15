import time
import logging
import json
import numpy as np
import soundfile as sf
from pathlib import Path
from collections import Counter
from components import (
    PhonemeRecognizer,
    IPAToARPAbetConverter,
    PhonemeToVisemeMapper
)


class Implementation1:
    """
    Baseline ground truth - offline phoneme recognition with overlapping windows
    
    Uses overlapping analysis to create a superior baseline compared to real-time processing:
    - 50% overlapping windows for redundancy
    - Consensus voting when phonemes appear in multiple windows
    - No real-time constraints
    - Processes all audio including silence
    """
    
    def __init__(self, sample_rate=16000, chunk_duration_ms=500, overlap_percent=50, silence_threshold=0.005):
        self.sample_rate = sample_rate
        self.chunk_duration_ms = chunk_duration_ms
        self.chunk_size = int(sample_rate * chunk_duration_ms / 1000)
        self.overlap_percent = overlap_percent
        self.hop_size = int(self.chunk_size * (1 - overlap_percent / 100))
        self.silence_threshold = silence_threshold
        
        # Initialize components
        self.recognizer = PhonemeRecognizer()
        self.ipa_converter = IPAToARPAbetConverter()
        self.mapper = PhonemeToVisemeMapper()
        
        # Storage for results
        self.results = []
        self.detection_map = {}  # Maps time positions to detected phonemes
        
        logging.info(f"Implementation 1 initialized (overlapping windows baseline)")
        logging.info(f"  Chunk size: {chunk_duration_ms}ms")
        logging.info(f"  Overlap: {overlap_percent}%")
        logging.info(f"  Hop size: {self.hop_size} samples ({self.hop_size/sample_rate*1000:.1f}ms)")
    
    def load_audio(self, audio_file_path):
        """Load audio from WAV file"""
        logging.info(f"Loading audio from: {audio_file_path}")
        
        try:
            audio_data, sample_rate = sf.read(audio_file_path)
            
            # Convert to mono if stereo
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            # Resample if necessary
            if sample_rate != self.sample_rate:
                logging.warning(f"Audio sample rate ({sample_rate}) differs from expected ({self.sample_rate})")
            
            duration = len(audio_data) / sample_rate
            logging.info(f"Loaded {duration:.2f} seconds of audio")
            
            return audio_data.astype(np.float32)
        
        except Exception as e:
            logging.error(f"Failed to load audio file: {e}")
            return None
    
    def process_chunk(self, chunk, chunk_start_time):
        """Process a single chunk and return detected phonemes with positions"""
        energy = float(np.abs(chunk).mean())
        
        # Still detect energy, but process anyway (we'll filter later if needed)
        is_silence = energy < self.silence_threshold
        
        # Recognize phonemes in this chunk
        try:
            ipa_phonemes = self.recognizer.recognize(chunk)
        except TypeError:
            ipa_phonemes = self.recognizer.recognize(chunk, self.sample_rate)
        
        results = []
        for ipa_phoneme in ipa_phonemes:
            # Handle multi-character sequences
            if len(ipa_phoneme) > 1:
                segments = self.ipa_converter.segment_ipa_string(ipa_phoneme)
                if len(segments) > 1:
                    for segment in segments:
                        arpabet = self.ipa_converter.convert(segment)
                        if arpabet and arpabet != '' and arpabet != 'SIL':
                            results.append({
                                'ipa': segment,
                                'arpabet': arpabet,
                                'time': chunk_start_time,
                                'energy': energy,
                                'is_silence': is_silence
                            })
                    continue
            
            # Normal single phoneme
            arpabet = self.ipa_converter.convert(ipa_phoneme)
            if arpabet and arpabet != '' and arpabet != 'SIL':
                results.append({
                    'ipa': ipa_phoneme,
                    'arpabet': arpabet,
                    'time': chunk_start_time,
                    'energy': energy,
                    'is_silence': is_silence
                })
        
        return results
    
    def merge_overlapping_detections(self, time_bin_ms=50):
        """
        Merge phoneme detections from overlapping windows using consensus
        
        Args:
            time_bin_ms: Time resolution for binning detections (ms)
        """
        if not self.detection_map:
            return []
        
        # Group detections by time bins
        time_bin_sec = time_bin_ms / 1000.0
        binned = {}
        
        for time_pos, phonemes in self.detection_map.items():
            bin_index = int(time_pos / time_bin_sec)
            if bin_index not in binned:
                binned[bin_index] = []
            binned[bin_index].extend(phonemes)
        
        # For each time bin, find consensus phoneme
        merged = []
        for bin_index in sorted(binned.keys()):
            phonemes = binned[bin_index]
            
            # Count occurrences
            arpabet_votes = Counter(p['arpabet'] for p in phonemes)
            most_common_arpabet, votes = arpabet_votes.most_common(1)[0]
            
            # Get a representative detection (prefer higher energy)
            representatives = [p for p in phonemes if p['arpabet'] == most_common_arpabet]
            best = max(representatives, key=lambda p: p['energy'])
            
            merged.append({
                'arpabet': most_common_arpabet,
                'ipa': best['ipa'],
                'time': bin_index * time_bin_sec,
                'energy': best['energy'],
                'confidence': votes / len(phonemes),  # What % of windows agreed
                'vote_count': votes,
                'total_detections': len(phonemes)
            })
        
        logging.info(f"Merged {len(self.detection_map)} raw detections into {len(merged)} consensus phonemes")
        return merged
    
    def analyze_offline(self, audio_file_path):
        """
        Analyze audio file offline with overlapping windows to establish ground truth
        """
        logging.info("=" * 60)
        logging.info("IMPLEMENTATION 1: Offline Analysis (Overlapping Windows)")
        logging.info("=" * 60)
        
        # Load audio from file
        audio_data = self.load_audio(audio_file_path)
        if audio_data is None:
            logging.error("Failed to load audio file")
            return None
        
        # Process audio with overlapping windows
        logging.info(f"Processing with {self.overlap_percent}% overlap...")
        logging.info(f"Window size: {self.chunk_duration_ms}ms, Hop: {self.hop_size/self.sample_rate*1000:.1f}ms")
        
        total_windows = int(np.ceil((len(audio_data) - self.chunk_size) / self.hop_size)) + 1
        logging.info(f"Total windows to process: {total_windows}")
        
        window_count = 0
        total_detections = 0
        
        start_time = time.time()
        
        # Sliding window analysis
        for i in range(0, len(audio_data) - self.chunk_size + 1, self.hop_size):
            chunk = audio_data[i:i + self.chunk_size]
            window_start_time = i / self.sample_rate
            
            # Process this window
            detections = self.process_chunk(chunk, window_start_time)
            
            # Add to detection map
            for det in detections:
                time_key = det['time']
                if time_key not in self.detection_map:
                    self.detection_map[time_key] = []
                self.detection_map[time_key].append(det)
                total_detections += 1
            
            window_count += 1
            
            if window_count % 100 == 0:
                logging.info(f"Processed {window_count}/{total_windows} windows...")
        
        recognition_time = time.time() - start_time
        
        logging.info(f"Recognition completed in {recognition_time:.2f}s")
        logging.info(f"Processed {window_count} overlapping windows")
        logging.info(f"Total raw detections: {total_detections}")
        
        # Merge overlapping detections using consensus
        logging.info("Merging overlapping detections...")
        merged = self.merge_overlapping_detections(time_bin_ms=50)
        
        # Convert to final results format
        for idx, phoneme_data in enumerate(merged):
            viseme = self.mapper.phoneme_to_viseme(phoneme_data['arpabet'])
            
            result = {
                'index': idx,
                'ipa_phoneme': phoneme_data['ipa'],
                'arpabet_phoneme': phoneme_data['arpabet'],
                'viseme': viseme,
                'recognition_time': phoneme_data['time'],
                'display_time': phoneme_data['time'],
                'energy': phoneme_data['energy'],
                'confidence': phoneme_data['confidence'],
                'vote_count': phoneme_data['vote_count'],
                'total_detections': phoneme_data['total_detections']
            }
            self.results.append(result)
        
        logging.info(f"Final ground truth: {len(self.results)} phonemes")
        logging.info(f"Average confidence: {np.mean([r['confidence'] for r in self.results]):.2%}")
        
        return self.results
    
    def get_results(self):
        """Get results for analysis"""
        if not self.results:
            return None
        
        # Calculate statistics
        intervals = []
        for i in range(1, len(self.results)):
            interval = (self.results[i]['recognition_time'] - self.results[i-1]['recognition_time']) * 1000
            intervals.append(interval)
        
        confidences = [r['confidence'] for r in self.results]
        
        stats = {
            'configuration': 'Implementation 1: Ground Truth (Overlapping Windows)',
            'phonemes_displayed': len(self.results),
            'mean_interval_ms': float(np.mean(intervals)) if intervals else 0,
            'std_interval_ms': float(np.std(intervals)) if intervals else 0,
            'mean_confidence': float(np.mean(confidences)),
            'min_confidence': float(np.min(confidences)),
            'phoneme_sequence': self.results
        }
        
        return stats
    
    def save_results(self, output_file="config1_results.json"):
        """Save results to file"""
        results = self.get_results()
        if results:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)
            logging.info(f"Results saved to {output_file}")


def run_implementation1(audio_file_path, output_file="config1_results.json"):
    """Run Implementation 1 complete workflow"""
    impl = Implementation1()
    
    # Analyze audio file offline
    impl.analyze_offline(audio_file_path)
    
    # Save results
    impl.save_results(output_file)
    
    return impl.get_results()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
        run_implementation1(audio_file)
    else:
        print("Usage: python implementation1.py <audio_file.wav>")
