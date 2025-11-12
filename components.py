import numpy as np
import cv2
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import sounddevice as sd
import wave
from scipy import signal
import logging
from pathlib import Path
import time
from collections import deque

class IPAToARPAbetConverter:
    """Converts IPA phoneme symbols to ARPAbet for viseme mapping (EXACT FROM Main.py)"""

    def __init__(self):
        # Comprehensive IPA to ARPAbet mapping
        self.ipa_to_arpabet = {
            # === VOWELS - FRONT ===
            # High front
            'i': 'IY',    'iː': 'IY',   'i̟': 'IY',
            'ɪ': 'IH',    'ɪ̈': 'IH',
            # Mid front
            'e': 'EY',    'eː': 'EY',   'ej': 'EY',   'ei': 'EY',  'eɪ': 'EY',
            'ɛ': 'EH',    'ɛː': 'EH',
            # Low front
            'æ': 'AE',    'æː': 'AE',   'a': 'AE',

            # === VOWELS - CENTRAL ===
            # High central
            'ɨ': 'IH',    'ʉ': 'UH',
            # Mid central
            'ə': 'AH',    'əː': 'AH',   'ɘ': 'AH',   'ɵ': 'AH',
            'ʌ': 'AH',    'ʌː': 'AH',
            'ɜ': 'ER',    'ɜː': 'ER',   'ɝ': 'ER',   'ɚ': 'ER',
            'ɐ': 'AH',    'ä': 'AA',
            # Syllabic consonants (treated as schwas)
            'ə̩': 'AH',   'ɚ̩': 'ER',

            # === VOWELS - BACK ===
            # High back
            'u': 'UW',    'uː': 'UW',   'ʊ': 'UH',   'ʊ̈': 'UH',
            # Mid back
            'o': 'OW',    'oː': 'OW',   'oʊ': 'OW',  'ou': 'OW',  'əʊ': 'OW',
            'ɔ': 'AO',    'ɔː': 'AO',   'ɔ̃': 'AO',
            # Low back
            'ɑ': 'AA',    'ɑː': 'AA',   'ɒ': 'AA',   'ɒː': 'AA',

            # === DIPHTHONGS ===
            'aɪ': 'AY',   'aj': 'AY',   'ai': 'AY',   'ɑɪ': 'AY',
            'aʊ': 'AW',   'aw': 'AW',   'au': 'AW',   'ɑʊ': 'AW',
            'ɔɪ': 'OY',   'oj': 'OY',   'oi': 'OY',   'ɔj': 'OY',

            # === CONSONANTS - STOPS ===
            'p': 'P',     'pʰ': 'P',
            'b': 'B',     'bʰ': 'B',
            't': 'T',     'tʰ': 'T',    'ʈ': 'T',
            'd': 'D',     'dʰ': 'D',    'ɖ': 'D',
            'k': 'K',     'kʰ': 'K',    'c': 'K',
            'g': 'G',     'gʰ': 'G',    'ɡ': 'G',    'ɢ': 'G',
            'ʔ': 'T',     # Glottal stop → treat as T

            # === CONSONANTS - FRICATIVES ===
            'f': 'F',     'ɸ': 'F',
            'v': 'V',     'β': 'V',
            'θ': 'TH',    'Θ': 'TH',  # Greek theta (different Unicode points)
            'ð': 'DH',    'Ð': 'DH',  # Eth (different cases)
            's': 'S',     'ɕ': 'S',
            'z': 'Z',     'ʑ': 'Z',
            'ʃ': 'SH',    'ʂ': 'SH',
            'ʒ': 'ZH',    'ʐ': 'ZH',
            'h': 'HH',    'ɦ': 'HH',    'ħ': 'HH',   'ʰ': 'HH',
            'x': 'K',     'χ': 'K',     'ɣ': 'G',

            # === CONSONANTS - AFFRICATES ===
            'tʃ': 'CH',   't͡ʃ': 'CH',   'ʧ': 'CH',   'tɕ': 'CH',
            'dʒ': 'JH',   'd͡ʒ': 'JH',   'ʤ': 'JH',   'dʑ': 'JH',

            # === CONSONANTS - NASALS ===
            'm': 'M',     'mʰ': 'M',    'ɱ': 'M',
            'n': 'N',     'nʰ': 'N',    'ɳ': 'N',    'n̩': 'N',  # Syllabic N
            'ŋ': 'NG',    'ɲ': 'NG',    'ɴ': 'NG',

            # === CONSONANTS - LIQUIDS ===
            'l': 'L',     'lʰ': 'L',    'ɫ': 'L',    'ɭ': 'L',   'l̩': 'L',  # Syllabic L
            'ɹ': 'R',     'r': 'R',     'ɾ': 'R',    'ʀ': 'R',   'ʁ': 'R',
            'ɻ': 'R',     'ɽ': 'R',

            # === CONSONANTS - GLIDES/APPROXIMANTS ===
            'w': 'W',     'ʍ': 'W',     'ʷ': 'W',    'ʋ': 'W',
            'j': 'Y',     'ʲ': 'Y',     'ɥ': 'Y',

            # === UPPERCASE (some models output these) ===
            'I': 'IH',    'E': 'EH',    'A': 'AH',    'O': 'OW',  'U': 'UW',
            'T': 'T',     'S': 'S',     'N': 'N',     'M': 'M',   'P': 'P',
            'B': 'B',     'D': 'D',     'K': 'K',     'G': 'G',   'F': 'F',
            'V': 'V',     'W': 'W',     'H': 'HH',    'L': 'L',   'R': 'R',
            'Y': 'Y',     'Z': 'Z',

            # === STRESS AND LENGTH MARKERS (ignore these) ===
            'ˈ': '',      'ˌ': '',      'ː': '',      'ˑ': '',
            '.': '',

            # === SPECIAL ===
            'SIL': 'SIL', 'SP': 'SIL',  '': 'SIL',    'sil': 'SIL',
        }

        logging.info("IPA-to-ARPAbet converter initialized")

    def segment_ipa_string(self, ipa_string):
        """Segment a multi-character IPA string into individual phonemes/diphthongs"""
        if not ipa_string:
            return []

        ipa_string = ipa_string.replace('͡', '')

        diphthongs = [
            'aɪə', 'aʊə', 'eɪə',
            'aɪ', 'aʊ', 'eɪ', 'oʊ', 'ɔɪ', 'əʊ', 'ɑɪ', 'ɑʊ', 'ɔj',
            'ai', 'au', 'ei', 'ou', 'oi', 'ej', 'aj', 'aw',
            'ɪə', 'ʊə', 'ɛə'
        ]

        affricates = ['tʃ', 'dʒ', 'ʧ', 'ʤ', 'tɕ', 'dʑ', 'ts', 'dz']

        segments = []
        i = 0

        while i < len(ipa_string):
            found = False

            for affricate in affricates:
                if ipa_string[i:i+len(affricate)] == affricate:
                    segments.append(affricate)
                    i += len(affricate)
                    found = True
                    break

            if found:
                continue

            for diphthong in diphthongs:
                if ipa_string[i:i+len(diphthong)] == diphthong:
                    segments.append(diphthong)
                    i += len(diphthong)
                    found = True
                    break

            if found:
                continue

            segments.append(ipa_string[i])
            i += 1

        return segments

    def convert(self, ipa_phoneme):
        """Convert an IPA phoneme to ARPAbet"""
        if not ipa_phoneme or ipa_phoneme.isspace():
            return 'SIL'

        ipa_clean = ipa_phoneme.strip()
        
        # Remove stress markers, length markers, and tie bars
        ipa_no_stress = ipa_clean.replace('ˈ', '').replace('ˌ', '').replace('ː', '').replace('.', '').replace('͡', '')

        # Try direct lookup first
        if ipa_clean in self.ipa_to_arpabet:
            result = self.ipa_to_arpabet[ipa_clean]
            if result:  # Not empty string (stress markers map to '')
                return result

        if ipa_no_stress and ipa_no_stress != ipa_clean:
            if ipa_no_stress in self.ipa_to_arpabet:
                result = self.ipa_to_arpabet[ipa_no_stress]
                if result:
                    return result

        # If already ARPAbet, return as-is
        ipa_upper = ipa_clean.upper()
        if ipa_upper in ['P', 'B', 'T', 'D', 'K', 'G', 'M', 'N', 'NG',
                         'F', 'V', 'TH', 'DH', 'S', 'Z', 'SH', 'ZH',
                         'CH', 'JH', 'HH', 'L', 'R', 'W', 'Y',
                         'IY', 'IH', 'EY', 'EH', 'AE', 'AA', 'AO', 'OW',
                         'UH', 'UW', 'AH', 'ER', 'AY', 'AW', 'OY', 'SIL']:
            return ipa_upper

        # Log unknown for debugging
        logging.warning(f"Unknown IPA symbol: '{ipa_phoneme}' -> defaulting to AH")
        return 'AH'


class PhonemeToVisemeMapper:
    """Maps phonemes to viseme categories using detailed 15-viseme system (EXACT FROM Main.py)"""

    def __init__(self):
        # Detailed phoneme-to-viseme mapping (15 viseme categories)
        # Based on articulatory phonetics and standard animation mappings
        self.mapping = {
            # Viseme 0: Silence/Rest
            'SIL': 'silence', 'SP': 'silence', '': 'silence',

            # Viseme 1: Bilabial stops and nasal (P, B, M)
            'P': 'bilabial_stop', 'B': 'bilabial_stop', 'M': 'bilabial_stop',

            # Viseme 2: Labiodental fricatives (F, V)
            'F': 'labiodental', 'V': 'labiodental',

            # Viseme 3: Dental fricatives (TH, DH)
            'TH': 'dental_fricative', 'DH': 'dental_fricative',

            # Viseme 4: Alveolar stops and nasal (T, D, N, L)
            'T': 'alveolar_stop', 'D': 'alveolar_stop',
            'N': 'alveolar_stop', 'L': 'alveolar_stop',

            # Viseme 5: Alveolar fricatives (S, Z)
            'S': 'alveolar_fricative', 'Z': 'alveolar_fricative',

            # Viseme 6: Post-alveolar (SH, ZH, CH, JH)
            'SH': 'postalveolar', 'ZH': 'postalveolar',
            'CH': 'postalveolar', 'JH': 'postalveolar',

            # Viseme 7: Velar (K, G, NG)
            'K': 'velar', 'G': 'velar', 'NG': 'velar',

            # Viseme 8: Glottal (H)
            'HH': 'glottal',

            # Viseme 9: Front Close (IY, IH, Y)
            'IY': 'front_close', 'IH': 'front_close', 'Y': 'front_close',

            # Viseme 10: Front Mid (EY, EH, AE)
            'EY': 'front_mid', 'EH': 'front_mid', 'AE': 'front_mid',

            # Viseme 11: Back Low (AA, AO, AY, AW)
            'AA': 'back_low', 'AO': 'back_low',
            'AY': 'back_low', 'AW': 'back_low',

            # Viseme 12: Close Rounded (UW, UH)
            'UW': 'rounded_close', 'UH': 'rounded_close',

            # Viseme 13: Mid Rounded (OW, OY, W)
            'OW': 'rounded_mid', 'OY': 'rounded_mid', 'W': 'rounded_mid',

            # Viseme 14: Central (AH, ER, R)
            'AH': 'central', 'ER': 'central', 'R': 'central',
        }

        # Human-readable descriptions for each viseme
        self.viseme_descriptions = {
            'silence': 'Silence/Rest - Mouth closed or neutral',
            'bilabial_stop': 'Bilabial (P,B,M) - Lips pressed together',
            'labiodental': 'Labiodental (F,V) - Teeth on lower lip',
            'dental_fricative': 'Dental (TH,DH) - Tongue between teeth',
            'alveolar_stop': 'Alveolar Stop (T,D,N,L) - Tongue to ridge',
            'alveolar_fricative': 'Alveolar Fricative (S,Z) - Teeth close',
            'postalveolar': 'Post-alveolar (SH,CH,JH) - Lips protruded',
            'velar': 'Velar (K,G,NG) - Back of tongue, open',
            'glottal': 'Glottal (H) - Open, relaxed',
            'front_close': 'Front Close (IY,IH,Y) - Spread lips, small opening',
            'front_mid': 'Front Mid (EY,EH,AE) - Spread lips, wider',
            'back_low': 'Back Low (AA,AO,AY,AW) - Large jaw drop',
            'rounded_close': 'Close Rounded (UW,UH) - Tight lip rounding, "boot"',
            'rounded_mid': 'Mid Rounded (OW,OY,W) - Open lip rounding, "go"',
            'central': 'Central (AH,ER,R) - Mid position',
        }

        # Log mapping statistics
        phoneme_count = len(self.mapping)
        viseme_count = len(set(self.mapping.values()))
        logging.info(f"Loaded phoneme-to-viseme mapping: {phoneme_count} phonemes -> {viseme_count} visemes")

    def phoneme_to_viseme(self, phoneme):
        """Convert phoneme to viseme category"""
        phoneme = phoneme.upper().strip()
        return self.mapping.get(phoneme, 'central')

    def get_viseme_description(self, viseme):
        """Get description of viseme articulation"""
        return self.viseme_descriptions.get(viseme, 'Unknown viseme')

    def get_all_visemes(self):
        """Return list of all viseme categories"""
        return list(self.viseme_descriptions.keys())


class PhonemeRecognizer:
    """Recognizes phonemes using pretrained wav2vec2 model (EXACT FROM Main.py)"""

    def __init__(self, model_name=None):
        # Phoneme recognition models
        self.model_options = [
            "bookbot/wav2vec2-ljspeech-gruut",
            "vitouphy/wav2vec2-xls-r-300m-timit-phoneme",
        ]

        if model_name:
            self.model_options.insert(0, model_name)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = None
        self.model = None
        self.model_name = None

        # Try loading models in order
        for model in self.model_options:
            try:
                logging.info(f"Loading phoneme model: {model}")

                import warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=FutureWarning)
                    warnings.filterwarnings("ignore", message="Some weights of.*were not initialized")

                    self.processor = Wav2Vec2Processor.from_pretrained(model)
                    self.model = Wav2Vec2ForCTC.from_pretrained(model).to(self.device)
                    self.model.eval()

                self.model_name = model
                logging.info(f"Model loaded successfully: {model}")
                logging.info(f"Running on: {self.device}")
                break

            except Exception as e:
                logging.warning(f"Could not load {model}: {str(e)[:150]}")
                continue

        if self.model is None:
            raise RuntimeError(
                "Failed to load phoneme recognition model. "
                "Please check your internet connection and ensure transformers library is installed."
            )

    def recognize(self, audio_chunk, sample_rate=16000):
        """Recognize phonemes from audio chunk"""
        try:
            # Prepare input
            audio_flat = audio_chunk.flatten()
            inputs = self.processor(
                audio_flat,
                sampling_rate=sample_rate,
                return_tensors="pt",
                padding=True
            )

            # Inference
            with torch.no_grad():
                logits = self.model(inputs.input_values.to(self.device)).logits

            # Decode
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self.processor.batch_decode(predicted_ids)[0]

            # Extract phonemes (space-separated in this model)
            phonemes = transcription.strip().split()

            # Filter out empty strings
            phonemes = [p for p in phonemes if p.strip()]

            return phonemes if phonemes else ['SIL']

        except Exception as e:
            logging.error(f"Recognition error: {e}")
            return ['SIL']


class VisemeVisualizer:
    """Displays viseme sprites in real-time (EXACT FROM Main.py)"""

    def __init__(self, sprite_dir='viseme_sprites', window_name='E2E-P2V Visualization'):
        self.window_name = window_name
        self.sprite_dir = Path(sprite_dir)
        self.current_viseme = 'silence'
        self.current_phoneme = ''
        self.sprites = {}
        self.display_width = 800
        self.display_height = 600
        self.needs_update = True  # Flag to track if display needs updating
        self.load_sprites()

    def load_sprites(self):
        """Load viseme sprite images (supports PNG and JPG)"""
        viseme_types = [
            'silence', 'bilabial_stop', 'labiodental', 'dental_fricative',
            'alveolar_stop', 'alveolar_fricative', 'postalveolar', 'velar',
            'glottal', 'front_close', 'front_mid', 'back_low',
            'rounded_close', 'rounded_mid', 'central'
        ]

        # Load sprites (check both PNG and JPG)
        for viseme in viseme_types:
            sprite_loaded = False

            # Try PNG first, then JPG
            for ext in ['.png', '.jpg', '.jpeg']:
                sprite_path = self.sprite_dir / f"{viseme}{ext}"
                if sprite_path.exists():
                    img = cv2.imread(str(sprite_path))
                    if img is not None:
                        self.sprites[viseme] = img
                        sprite_loaded = True
                        logging.info(f"Loaded {viseme}{ext}")
                        break

            # Create placeholder only if no image found
            if not sprite_loaded:
                logging.warning(f"No image found for {viseme}, creating placeholder")
                if not self.sprite_dir.exists():
                    self.sprite_dir.mkdir(exist_ok=True)
                self.sprites[viseme] = self.create_placeholder_image(viseme)
                # Save placeholder as PNG
                cv2.imwrite(str(self.sprite_dir / f"{viseme}.png"), self.sprites[viseme])

        logging.info(f"Loaded {len(self.sprites)} viseme sprites")

    def create_placeholder_image(self, viseme):
        """Create a simple colored placeholder for missing visemes"""
        img = np.zeros((400, 400, 3), dtype=np.uint8)

        # Simple color scheme
        color = (128, 128, 128)  # Gray default

        cv2.circle(img, (200, 200), 150, color, -1)

        # Add text
        words = viseme.split('_')
        if len(words) == 1:
            cv2.putText(img, viseme.upper(), (50, 200),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            y_offset = 180
            for word in words:
                cv2.putText(img, word.upper(), (70, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                y_offset += 35

        return img

    def update_display(self, viseme, phoneme):
        """Update the displayed viseme sprite (only if changed)"""
        if self.current_viseme != viseme or self.current_phoneme != phoneme:
            self.current_viseme = viseme
            self.current_phoneme = phoneme
            self.needs_update = True  # Mark that we need to re-render

    def render(self):
        """Render current frame (only if changed)"""
        if not self.needs_update:
            return False  # No update needed

        # Get sprite
        sprite = self.sprites.get(self.current_viseme, self.sprites.get('silence', self.sprites.get('central')))

        # Resize sprite to fit display
        sprite_resized = cv2.resize(sprite, (self.display_width, self.display_height - 100))

        # Create display frame
        frame = np.zeros((self.display_height, self.display_width, 3), dtype=np.uint8)
        frame[0:self.display_height - 100] = sprite_resized

        # Add text overlay with viseme name (format for readability)
        viseme_display = self.current_viseme.replace('_', ' ').title()
        info_text = f"Viseme: {viseme_display}  |  Phoneme: {self.current_phoneme}"
        cv2.putText(frame, info_text, (20, self.display_height - 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Add instructions
        cv2.putText(frame, "Press 'Q' to quit", (20, self.display_height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow(self.window_name, frame)
        self.needs_update = False  # Reset flag after rendering
        return True  # Update was performed

    def close(self):
        """Close visualization window"""
        cv2.destroyAllWindows()


class AudioRecorder:
    """Records audio to WAV file"""

    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.channels = 1
        self.recording = False
        self.audio_buffer = []

    def record_until_keypress(self, output_path):
        """Record audio until user presses Enter"""
        import sys
        import threading
        
        logging.info("Recording audio - press ENTER to stop...")
        print("\n" + "="*60)
        print("🎤 RECORDING IN PROGRESS")
        print("="*60)
        print("Speak clearly into your microphone...")
        print("\nPress ENTER when you're finished speaking")
        print("="*60 + "\n")
        
        self.recording = True
        self.audio_buffer = []
        
        # Start recording in a callback
        def audio_callback(indata, frames, time_info, status):
            if status:
                logging.warning(f"Audio status: {status}")
            if self.recording:
                self.audio_buffer.append(indata.copy())
        
        # Start the audio stream
        stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype='float32',
            callback=audio_callback
        )
        
        with stream:
            # Wait for Enter key
            input()  # Blocks until user presses Enter
            self.recording = False
        
        # Concatenate all recorded chunks
        if self.audio_buffer:
            audio_data = np.concatenate(self.audio_buffer, axis=0)
            duration = len(audio_data) / self.sample_rate
            
            print(f"\n✓ Recording stopped")
            print(f"  Duration: {duration:.1f} seconds")
            print(f"  Saving to: {output_path}\n")
            
            # Save to WAV file
            self._save_wav(audio_data, output_path)
            logging.info(f"Audio saved to {output_path} ({duration:.1f}s)")
            
            return output_path, duration
        else:
            logging.error("No audio recorded")
            return None, 0

    def _save_wav(self, audio_data, output_path):
        """Save audio data to WAV file"""
        # Convert to int16
        audio_int16 = (audio_data * 32767).astype(np.int16)
        
        with wave.open(str(output_path), 'w') as wav_file:
            wav_file.setnchannels(self.channels)
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(audio_int16.tobytes())


class AudioFilePlayer:
    """Plays audio from WAV file with precise timing"""

    def __init__(self, file_path):
        self.file_path = file_path
        self.sample_rate = None
        self.audio_data = None
        self._load_audio()

    def _load_audio(self):
        """Load audio file"""
        with wave.open(str(self.file_path), 'r') as wav_file:
            self.sample_rate = wav_file.getframerate()
            frames = wav_file.readframes(wav_file.getnframes())
            self.audio_data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

    def play(self):
        """Play the audio file"""
        sd.play(self.audio_data, self.sample_rate)
        
    def stop(self):
        """Stop playback"""
        sd.stop()
        
    def wait(self):
        """Wait for playback to finish"""
        sd.wait()
        
    def get_duration(self):
        """Get duration in seconds"""
        return len(self.audio_data) / self.sample_rate


class AudioFileStreamer:
    """Streams audio from file in chunks (simulating real-time)"""

    def __init__(self, file_path, chunk_duration_ms=500):
        self.file_path = file_path
        self.chunk_duration_ms = chunk_duration_ms
        self.sample_rate = None
        self.audio_data = None
        self.current_position = 0
        self._load_audio()

    def _load_audio(self):
        """Load audio file"""
        with wave.open(str(self.file_path), 'r') as wav_file:
            self.sample_rate = wav_file.getframerate()
            frames = wav_file.readframes(wav_file.getnframes())
            self.audio_data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

    def get_next_chunk(self):
        """Get next audio chunk"""
        chunk_size = int(self.sample_rate * self.chunk_duration_ms / 1000)
        
        if self.current_position >= len(self.audio_data):
            return None
            
        chunk = self.audio_data[self.current_position:self.current_position + chunk_size]
        self.current_position += chunk_size
        
        return chunk

    def reset(self):
        """Reset to beginning"""
        self.current_position = 0

    def has_more(self):
        """Check if more audio available"""
        return self.current_position < len(self.audio_data)