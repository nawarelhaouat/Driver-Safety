# =============================================================================
# generate_alarm.py  (script utilitaire — à exécuter une seule fois)
# -----------------------------------------------------------------------------
# Génère un fichier alarm.wav simple (bip répété) dans le dossier assets/.
# Utile si vous n'avez pas de fichier audio à disposition.
#
# Usage : python generate_alarm.py
# =============================================================================

import numpy as np
import wave
import struct
import os

def generate_beep_wav(output_path, frequency=880, duration=0.5, sample_rate=44100, volume=0.7):
    """
    Génère un son sinusoïdal (bip) et le sauvegarde en .wav.

    Paramètres :
        output_path  : chemin de sortie du fichier .wav
        frequency    : fréquence en Hz (880 Hz = La5, très audible)
        duration     : durée du bip en secondes
        sample_rate  : fréquence d'échantillonnage (44100 Hz = qualité CD)
        volume       : volume entre 0.0 et 1.0
    """
    # Nombre total d'échantillons
    n_samples = int(sample_rate * duration)

    # Génération d'une onde sinusoïdale
    t = np.linspace(0, duration, n_samples, endpoint=False)
    wave_data = np.sin(2 * np.pi * frequency * t)

    # Application d'une enveloppe (fade in / fade out) pour éviter les clics
    fade_len = int(sample_rate * 0.01)  # 10ms de fade
    fade_in  = np.linspace(0, 1, fade_len)
    fade_out = np.linspace(1, 0, fade_len)
    wave_data[:fade_len]   *= fade_in
    wave_data[-fade_len:]  *= fade_out

    # Mise à l'échelle sur 16 bits (entier signé)
    wave_data = (wave_data * volume * 32767).astype(np.int16)

    # Écriture du fichier WAV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with wave.open(output_path, 'w') as wf:
        wf.setnchannels(1)          # Mono
        wf.setsampwidth(2)          # 16 bits = 2 octets
        wf.setframerate(sample_rate)
        wf.writeframes(wave_data.tobytes())

    print(f"[OK] Fichier généré : {output_path}")


if __name__ == "__main__":
    # Dossier du script
    base_dir   = os.path.dirname(os.path.abspath(__file__))
    alarm_path = os.path.join(base_dir, "assets", "alarm.wav")

    # Génération : 3 bips de 0.5s à 880 Hz séparés par 0.1s de silence
    sample_rate = 44100
    beep_data   = np.array([], dtype=np.int16)

    for _ in range(3):
        # Bip
        duration  = 0.5
        n_samples = int(sample_rate * duration)
        t         = np.linspace(0, duration, n_samples, endpoint=False)
        bip       = np.sin(2 * np.pi * 880 * t)

        # Fade
        fade_len = int(sample_rate * 0.01)
        bip[:fade_len]  *= np.linspace(0, 1, fade_len)
        bip[-fade_len:] *= np.linspace(1, 0, fade_len)

        bip = (bip * 0.7 * 32767).astype(np.int16)

        # Silence entre les bips (100ms)
        silence = np.zeros(int(sample_rate * 0.1), dtype=np.int16)

        beep_data = np.concatenate([beep_data, bip, silence])

    # Sauvegarde
    os.makedirs(os.path.join(base_dir, "assets"), exist_ok=True)
    with wave.open(alarm_path, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(beep_data.tobytes())

    print(f"[OK] alarm.wav généré dans : {alarm_path}")
    print("     Durée totale : ~1.8 secondes (3 bips)")
