import os
import base64
import requests
from elevenlabs.client import ElevenLabs
from elevenlabs import Voice, VoiceSettings
from dotenv import load_dotenv
import soundfile as sf
import numpy as np
import torch
from typing import Optional
import logging
from pathlib import Path
import re

load_dotenv()

# Silence the fork/parallelism warning from HF tokenizers
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Configuration from .env
class Settings:
    def __init__(self):
        self.MODEL_PATH = os.getenv("MODEL_PATH", "microsoft/VibeVoice-1.5B")
        self.DEVICE = os.getenv("DEVICE", "cpu")
        self.CFG_SCALE = float(os.getenv("CFG_SCALE", 1.1))
        self.SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", 16000))
        self.VOICES_DIR = Path("voices")  # Directory for voice samples

settings = Settings()
settings.VOICES_DIR.mkdir(exist_ok=True)

# VoiceProfile and VoiceType classes
class VoiceProfile:
    def __init__(self, id: str, name: str, type: str, file_path: str):
        self.id = id
        self.name = name
        self.type = type
        self.file_path = file_path

class VoiceType:
    PRESET = "PRESET"
    UPLOADED = "UPLOADED"

# VoiceService class (adapted from vibevoice-studio)
class VoiceService:
    """Service for voice synthesis operations."""

    def __init__(self):
        self.model = None
        self.processor = None
        self.voices_cache = {}
        self.model_loaded = False
        self._initialize_model()
        self._load_voices()

    def _initialize_model(self):
        try:
            from vibevoice.modular.modeling_vibevoice_inference import (
                VibeVoiceForConditionalGenerationInference,
            )
            from vibevoice.processor.vibevoice_processor import (
                VibeVoiceProcessor,
            )
        except ImportError:
            logging.error(
                "VibeVoice not installed. Install with:\n"
                "  git clone https://github.com/microsoft/VibeVoice.git\n"
                "  cd VibeVoice && pip install -e ."
            )
            return

        # Decide device
        use_cuda = settings.DEVICE == "cuda" and torch.cuda.is_available()
        use_mps = (
            settings.DEVICE == "mps"
            and getattr(torch.backends, "mps", None) is not None
            and torch.backends.mps.is_available()
        )
        torch_device = "cuda" if use_cuda else ("mps" if use_mps else "cpu")
        dtype = torch.float16 if use_cuda else torch.float32

        logging.info(
            f"Loading model from {settings.MODEL_PATH} "
            f"on device={torch_device} dtype={dtype}"
        )

        # Load processor (no token to avoid tokenizer issues)
        self.processor = VibeVoiceProcessor.from_pretrained(settings.MODEL_PATH)

        # Load model
        load_kwargs = {"torch_dtype": dtype, "token": os.getenv("HUGGINGFACE_HUB_TOKEN")}
        if use_cuda:
            load_kwargs["attn_implementation"] = "flash_attention_2"

        try:
            self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                settings.MODEL_PATH,
                **load_kwargs,
            )
        except Exception as e1:
            logging.warning(
                f"Primary load failed ({e1}). Retrying without attn_implementation."
            )
            load_kwargs.pop("attn_implementation", None)
            self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                settings.MODEL_PATH,
                **load_kwargs,
            )

        self.model.to(torch_device)
        self.model.eval()
        try:
            self.model.set_ddpm_inference_steps(num_steps=10)
        except Exception:
            pass

        self.model_loaded = True
        logging.info("Model loaded successfully.")

    def _load_voices(self):
        voice_files = []
        for ext in ("*.wav", "*.mp3", "*.m4a", "*.flac", "*.ogg"):
            voice_files.extend(settings.VOICES_DIR.glob(ext))

        if not voice_files:
            logging.info(
                "No voice files found in voices/. Using placeholder for narration."
            )
            placeholder_path = self._create_placeholder_voice("default_host")
            voice_id = "default_host"  # Match voice_id to filename
            profile = VoiceProfile(
                id=voice_id,
                name="Default Host Voice",
                type=VoiceType.PRESET,
                file_path=placeholder_path,
            )
            self.voices_cache[voice_id] = profile
            logging.info("Loaded placeholder voice: Default Host")

        for voice_file in voice_files:
            voice_id = str(voice_file.stem)
            profile = VoiceProfile(
                id=voice_id,
                name=voice_file.stem,
                type=VoiceType.PRESET,
                file_path=str(voice_file),
            )
            self.voices_cache[voice_id] = profile
            logging.info(f"Loaded voice: {voice_file.stem}")

    def _create_placeholder_voice(self, name: str) -> str:
        duration = 5.0
        sr = settings.SAMPLE_RATE
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)
        audio = np.clip(audio, -1.0, 1.0)
        path = settings.VOICES_DIR / f"{name}.wav"
        sf.write(path, audio, sr)
        return str(path)

    def generate_speech(
        self,
        text: str,
        voice_id: str = "default_host",  # Updated to match _load_voices
        num_speakers: int = 1,
        cfg_scale: float = 1.3,
    ) -> Optional[np.ndarray]:
        try:
            voice_profile = self.voices_cache.get(voice_id)
            if not voice_profile:
                raise ValueError(f"Voice profile {voice_id} not found")

            if not (self.model_loaded and self.model and self.processor):
                logging.warning("Model not loaded — returning sample placeholder audio.")
                return self._generate_sample_audio(text)

            logging.info(f"Generating speech with voice: {voice_profile.name}")
            logging.info(f"Text input: {text[:100]}...")  # Log first 100 chars

            formatted_text = self._format_text_for_speakers(text, num_speakers)
            logging.info(f"Formatted text: {formatted_text[:100]}...")

            inputs = self.processor(
                text=[formatted_text],
                voice_samples=[[voice_profile.file_path]],
                padding=True,
                return_tensors="pt",
                return_attention_mask=True,
            )

            model_device = next(self.model.parameters()).device
            for k, v in list(inputs.items()):
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(model_device)

            logging.info("Starting generation...")

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=None,
                cfg_scale=cfg_scale,
                tokenizer=self.processor.tokenizer,
                generation_config={"do_sample": False},
                verbose=False,
            )

            if (
                getattr(outputs, "speech_outputs", None)
                and outputs.speech_outputs[0] is not None
            ):
                audio_tensor = outputs.speech_outputs[0]
                if audio_tensor.dtype != torch.float32:
                    audio_tensor = audio_tensor.to(torch.float32)
                audio_array = audio_tensor.detach().cpu().numpy()
                audio_array = np.clip(audio_array, -1.0, 1.0)
                logging.info(f"Generated audio shape: {audio_array.shape}")
                return audio_array

            logging.error("No speech output generated by the model.")
            return None

        except Exception as e:
            logging.error(f"Speech generation error: {e}", exc_info=True)
            return self._generate_sample_audio(text)

    def _format_text_for_speakers(self, text: str, num_speakers: int) -> str:
        if num_speakers <= 1:
            if not text.strip().startswith("Speaker") and not text.strip().startswith("Host:"):
                return f"Host: {text}"
            return text

        lines = [ln.strip() for ln in text.splitlines()]
        formatted = []
        current = 0
        for ln in lines:
            if not ln:
                continue
            if ln.startswith("Speaker") or ln.startswith("Host:"):
                formatted.append(ln)
            else:
                formatted.append(f"Speaker {current}: {ln}")
                current = (current + 1) % num_speakers
        return "\n".join(formatted)

    def _generate_sample_audio(self, text: str) -> np.ndarray:
        duration = float(min(10.0, max(1.0, len(text) * 0.05)))
        sr = settings.SAMPLE_RATE
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        freqs = [220.0, 440.0, 660.0]
        audio = sum(
            0.25 / (i + 1) * np.sin(2 * np.pi * f * t) for i, f in enumerate(freqs)
        )
        env = np.minimum(1.0, np.linspace(0, 1.0, len(t)) * 3.0) * np.exp(-t * 0.6)
        audio = (audio * env).astype(np.float32)
        peak = float(np.max(np.abs(audio))) if audio.size else 1.0
        if peak > 0:
            audio = 0.8 * (audio / peak)
        return audio

import re

def split_text(text: str, max_length: int = 500) -> list[str]:
    """
    Splits text into chunks for TTS:
    - Splits on sentence-ending punctuation (.?!).
    - Respects max_length without breaking words.
    - Avoids splitting inside numbers, abbreviations, URLs, and emails.
    """
    # Protect decimals, abbreviations, and URLs/emails before splitting
    protected = {
        r'(?<=\d)\.(?=\d)': '<DECIMAL>',
        r'(?<=\b[A-Z])\.(?=[A-Z]\b)': '<ABBR>',
        r'https?://\S+': lambda m: m.group(0).replace('.', '<DOT>'),
        r'\S+@\S+': lambda m: m.group(0).replace('.', '<DOT>')
    }

    safe_text = text
    for pattern, repl in protected.items():
        safe_text = re.sub(pattern, repl, safe_text)

    # Split on sentence-ending punctuation followed by space/newline
    sentences = re.split(r'(?<=[.!?])\s+', safe_text.strip())

    # Restore protected markers
    def restore(s: str) -> str:
        return (
            s.replace('<DECIMAL>', '.')
             .replace('<ABBR>', '.')
             .replace('<DOT>', '.')
        )

    chunks = []
    current_chunk = []
    current_len = 0

    for sentence in sentences:
        sentence = restore(sentence.strip())
        if not sentence:
            continue

        if current_len + len(sentence) <= max_length:
            current_chunk.append(sentence)
            current_len += len(sentence) + 1
        else:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_len = len(sentence) + 1

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def generate_audio_vibevoice(text, output_file="podcast_vibevoice.wav"):
    """
    Generate audio using Microsoft VibeVoice model.
    Falls back to placeholder sine wave if the model fails.
    """
    try:
        voice_service = VoiceService()
        if not voice_service.model_loaded:
            print("❌ VibeVoice model failed to load. Check logs.")
            return

        audio_array = voice_service.generate_speech(
            text=text,
            voice_id="default_host",  # Match _load_voices
            num_speakers=1,
            cfg_scale=settings.CFG_SCALE
        )

        if audio_array is not None and len(audio_array) > 0:
            # --- Fix shape issues ---
            if audio_array.ndim == 2 and audio_array.shape[0] == 1:
                audio_array = audio_array.squeeze(0)  # (N,)
            elif audio_array.ndim == 2 and audio_array.shape[1] == 1:
                audio_array = audio_array.squeeze(1)  # (N,)

            # --- Normalize to [-1, 1] just in case ---
            peak = np.max(np.abs(audio_array)) if np.max(np.abs(audio_array)) > 0 else 1.0
            audio_array = np.clip(audio_array / peak, -1.0, 1.0).astype(np.float32)

            print(f"Audio dtype: {audio_array.dtype}, shape: {audio_array.shape}")

            # --- Save to file ---
            sf.write(output_file, audio_array, settings.SAMPLE_RATE)
            print(f"✅ Podcast audio saved as {output_file}")
        else:
            print("❌ No audio generated. Check text format or voice sample.")

    except Exception as e:
        print(f"❌ VibeVoice generation failed: {e}")

import io

def generate_audio_sarvam(text, output_file="podcast_sarvam.wav"):
    """
    Generate audio using Sarvam TTS.
    Handles API limit of 3 chunks per request by batching.
    Properly concatenates audio as WAV.
    """
    import soundfile as sf

    api_key = os.getenv("SARVAM_API_KEY")
    if not api_key:
        print("❌ SARVAM_API_KEY not set.")
        return

    text_chunks = split_text(text)
    headers = {
        "api-subscription-key": api_key,
        "Content-Type": "application/json"
    }

    final_audio = []
    sr = None  # sample rate

    # Batch chunks in groups of 3
    for i in range(0, len(text_chunks), 3):
        batch = text_chunks[i:i+3]
        payload = {
            "speaker": "anushka",
            "target_language_code": "en-IN",
            "inputs": batch,
            "pitch": 1,
            "pace": 1,
            "loudness": 1,
            "speech_sample_rate": 22050,
            "enable_preprocessing": True,
            "model": "bulbul:v2"
        }

        try:
            response = requests.post(
                "https://api.sarvam.ai/text-to-speech",
                json=payload,
                headers=headers
            )

            if response.status_code == 200:
                audio_b64_list = response.json().get("audios", [])
                if not audio_b64_list:
                    print(f"❌ No audio returned for batch {i//3 + 1}.")
                    continue

                for idx, b64 in enumerate(audio_b64_list):
                    audio_bytes = base64.b64decode(b64)
                    audio_stream = io.BytesIO(audio_bytes)
                    audio_data, sr = sf.read(audio_stream, dtype="float32")
                    final_audio.append(audio_data)

                print(f"✅ Processed batch {i//3 + 1}/{(len(text_chunks)+2)//3}")

            else:
                print(f"❌ Sarvam API Error {response.status_code}: {response.text}")
        except Exception as e:
            print(f"❌ Sarvam TTS error in batch {i//3 + 1}: {e}")

    # Save final stitched audio
    if final_audio and sr:
        full_audio = np.concatenate(final_audio, axis=0)
        sf.write(output_file, full_audio, sr)
        print(f"🎙️ Full podcast audio saved as {output_file}")
    else:
        print("❌ No audio generated at all.")


def generate_audio_elevenlabs(text, output_file="podcast_elevenlabs.wav"):
    """
    Generate audio using ElevenLabs TTS.
    """
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        print("❌ ELEVENLABS_API_KEY not set.")
        return

    try:
        client = ElevenLabs(api_key=api_key)

        voice_id = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")  # default "Rachel"

        audio_stream = client.text_to_speech.convert(
            voice_id=voice_id,
            model_id="eleven_monolingual_v1",
            text=text,
            voice_settings=VoiceSettings(stability=0.5, similarity_boost=0.75)
        )

        with open(output_file, "wb") as f:
            for chunk in audio_stream:
                f.write(chunk)

        print(f"✅ Podcast audio saved as {output_file}")

    except Exception as e:
        print(f"❌ ElevenLabs TTS error: {e}")

def main():
    summary_file = "podcast_summary.txt"
    if not os.path.exists(summary_file):
        print(f"❌ {summary_file} not found.")
        return
    with open(summary_file, "r", encoding="utf-8") as f:
        text = f.read().strip()
    if not text:
        print("❌ No content in summary.")
        return

    # generate_audio_sarvam(text)
    # generate_audio_elevenlabs(text)
    # Test with short text
    test_text = "Speaker 1: Hello, this is a test."
    print("Testing with short text...")
    generate_audio_vibevoice(test_text, "test_vibevoice.wav")
    # print("Testing with summary text...")
    # print(text)
    # generate_audio_vibevoice(text)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()