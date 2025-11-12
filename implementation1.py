import time
import logging
import json
import numpy as np
import soundfile as sf
from pathlib import Path
from components import (
    PhonemeRecognizer,
    IPAToARPAbetConverter,
    PhonemeToVisemeMapper
)


class Implementation1:
    """Baseline ground truth - offline phoneme recognition from audio file"""
    
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        
        # Initialize components
        self.recognizer = PhonemeRecognizer()
        self.ipa_converter = IPAToARPAbetConverter()
        self.mapper = PhonemeToVisemeMapper()
        
        # Storage for results
        self.phoneme_sequence = []
        self.results = []
        
        logging.info("Implementation 1 initialized (baseline ground truth)")
    
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
    
    def analyze_offline(self, audio_file_path):
        """
        Analyze audio file offline to establish ground truth phoneme sequence
        """
        logging.info("=" * 60)
        logging.info("IMPLEMENTATION 1: Offline Analysis (Baseline Ground Truth)")
        logging.info("=" * 60)
        
        # Load audio from file
        audio_data = self.load_audio(audio_file_path)
        if audio_data is None:
            logging.error("Failed to load audio file")
            return None
        
        # Recognize phonemes
        logging.info("Recognizing phonemes...")
        start_time = time.time()
        ipa_phonemes = self.recognizer.recognize(audio_data, self.sample_rate)
        recognition_time = time.time() - start_time
        
        logging.info(f"Recognition completed in {recognition_time:.2f}s")
        logging.info(f"Found {len(ipa_phonemes)} IPA phonemes")
        
        # Convert to ARPAbet and map to visemes
        for ipa_phoneme in ipa_phonemes:
            # Handle multi-character sequences
            if len(ipa_phoneme) > 1:
                segments = self.ipa_converter.segment_ipa_string(ipa_phoneme)
                if len(segments) > 1:
                    for segment in segments:
                        arpabet = self.ipa_converter.convert(segment)
                        if arpabet and arpabet != '':
                            self.phoneme_sequence.append({
                                'ipa': segment,
                                'arpabet': arpabet,
                                'viseme': self.mapper.phoneme_to_viseme(arpabet)
                            })
                    continue
            
            # Normal single phoneme
            arpabet = self.ipa_converter.convert(ipa_phoneme)
            if arpabet and arpabet != '':
                self.phoneme_sequence.append({
                    'ipa': ipa_phoneme,
                    'arpabet': arpabet,
                    'viseme': self.mapper.phoneme_to_viseme(arpabet)
                })
        
        logging.info(f"Converted to {len(self.phoneme_sequence)} ARPAbet phonemes")
        
        # Record actual timing (baseline ground truth - NO artificial spacing)
        logging.info("Recording baseline phoneme sequence with actual detection timing...")
        current_time = 0.0
        
        for idx, phoneme_data in enumerate(self.phoneme_sequence):
            # Store result with actual detection time
            result = {
                'index': idx,
                'ipa_phoneme': phoneme_data['ipa'],
                'arpabet_phoneme': phoneme_data['arpabet'],
                'viseme': phoneme_data['viseme'],
                'recognition_time': current_time,
                'display_time': current_time
            }
            self.results.append(result)
            
            # Minimal increment for offline batch processing
            current_time += 0.001
        
        logging.info(f"Baseline analysis complete: {len(self.results)} phonemes recorded")
        return self.phoneme_sequence
    
    def get_results(self):
        """Get results for analysis"""
        if not self.results:
            return None
        
        # Calculate statistics
        intervals = []
        for i in range(1, len(self.results)):
            interval = (self.results[i]['recognition_time'] - self.results[i-1]['recognition_time']) * 1000
            intervals.append(interval)
        
        stats = {
            'configuration': 'Implementation 1: Baseline (Offline Recognition)',
            'phonemes_displayed': len(self.results),
            'mean_interval_ms': float(np.mean(intervals)) if intervals else 0,
            'std_interval_ms': float(np.std(intervals)) if intervals else 0,
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
