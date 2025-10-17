
import argparse, json, os, pathlib, random
from pathlib import Path
import pyttsx3
from pydub import AudioSegment

def tts_to_wav(text, path_wav, voice_name=None, rate=175):
    engine = pyttsx3.init()
    if voice_name:
        for v in engine.getProperty('voices'):
            if voice_name.lower() in (v.id.lower() + " " + v.name.lower()):
                engine.setProperty('voice', v.id)
                break
    engine.setProperty('rate', rate)
    engine.save_to_file(text, str(path_wav))
    engine.runAndWait()

def mix_with_noise(speech_path, noise_dir, out_path, snr_db=10):
    speech = AudioSegment.from_file(speech_path)
    noise_files = [p for p in Path(noise_dir).glob("*") if p.suffix.lower() in [".wav",".mp3",".flac",".ogg",".m4a"]]
    if not noise_files:
        speech.export(out_path, format="wav")
        return
    noise = AudioSegment.from_file(random.choice(noise_files)).set_channels(speech.channels).set_frame_rate(speech.frame_rate)
    # Loop or trim noise to match length
    if len(noise) < len(speech):
        times = len(speech) // len(noise) + 1
        noise = noise * times
    noise = noise[:len(speech)]
    # Adjust noise level for desired SNR
    # SNR(dB) ≈ Pspeech_dBFS - Pnoise_dBFS
    noise = noise - (noise.dBFS - (speech.dBFS - snr_db))
    mixed = speech.overlay(noise)
    mixed.export(out_path, format="wav")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="augmented.jsonl")
    ap.add_argument("--audio-out", required=True, help="output directory for audio")
    ap.add_argument("--noise-dir", required=True, help="directory with noise audio files (optional)")
    ap.add_argument("--snr", type=int, default=10)
    args = ap.parse_args()

    Path(args.audio_out).mkdir(parents=True, exist_ok=True)

    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            uid = ex["id"]
            text = ex["command"]
            clean_wav = Path(args.audio_out) / f"{uid}_clean.wav"
            noisy_wav = Path(args.audio_out) / f"{uid}_noisy.wav"
            if not clean_wav.exists():
                tts_to_wav(text, clean_wav)
            mix_with_noise(clean_wav, args.noise_dir, noisy_wav, snr_db=args.snr)
    print(f"Audio saved to {args.audio_out}")

if __name__ == "__main__":
    main()
