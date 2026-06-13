# =============================================================================
# src/alarm_manager.py
# -----------------------------------------------------------------------------
# Gestion de l'alarme sonore — compatible pygame ET pygame-ce.
# =============================================================================

import os
import threading

# Import pygame ou pygame-ce (fork compatible, même API)
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    try:
        import pygame_ce as pygame
        PYGAME_AVAILABLE = True
    except ImportError:
        PYGAME_AVAILABLE = False
        print("[AlarmManager] pygame/pygame-ce non disponible — alarme désactivée.")


class AlarmManager:

    def __init__(self, alarm_path):
        self.alarm_path   = alarm_path
        self.is_playing   = False
        self._initialized = False

        if PYGAME_AVAILABLE:
            try:
                pygame.mixer.init()
                self._initialized = True
            except Exception as e:
                print(f"[AlarmManager] Erreur init pygame : {e}")

    def play(self):
        if self.is_playing:
            return
        if not PYGAME_AVAILABLE or not self._initialized:
            print("[ALERTE] Conducteur somnolent !")
            self.is_playing = True
            return
        if not os.path.exists(self.alarm_path):
            print(f"[AlarmManager] Fichier audio introuvable : {self.alarm_path}")
            self.is_playing = True
            return
        thread = threading.Thread(target=self._play_sound, daemon=True)
        thread.start()
        self.is_playing = True

    def _play_sound(self):
        try:
            pygame.mixer.music.load(self.alarm_path)
            pygame.mixer.music.play(-1)
            while self.is_playing:
                pygame.time.Clock().tick(10)
            pygame.mixer.music.stop()
        except Exception as e:
            print(f"[AlarmManager] Erreur lecture : {e}")

    def stop(self):
        if not self.is_playing:
            return
        self.is_playing = False
        if PYGAME_AVAILABLE and self._initialized:
            try:
                pygame.mixer.music.stop()
            except Exception:
                pass

    def release(self):
        self.stop()
        if PYGAME_AVAILABLE and self._initialized:
            try:
                pygame.mixer.quit()
            except Exception:
                pass