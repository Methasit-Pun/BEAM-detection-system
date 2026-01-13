"""
Alert System Module
Handles audio and visual alerts for violations
"""

import logging
import os
import threading
import time

try:
    import pyaudio
    import wave
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    
try:
    from playsound import playsound
    PLAYSOUND_AVAILABLE = True
except ImportError:
    PLAYSOUND_AVAILABLE = False


class AlertSystem:
    """Manage audio and visual alerts"""
    
    def __init__(self, config):
        """
        Initialize alert system
        
        Args:
            config: Configuration dictionary with alert settings
        """
        self.config = config
        self.enabled = config.get('enabled', True)
        self.audio_file = config.get('audio_file', 'audio/violation_alert.wav')
        self.visual_overlay = config.get('visual_overlay', True)
        self.overlay_color = tuple(config.get('overlay_color', [0, 0, 255]))  # BGR
        self.overlay_duration = config.get('overlay_duration', 2.0)
        
        self.is_playing = False
        self.last_alert_time = 0
        self.logger = logging.getLogger(__name__)
        
        self._check_audio_availability()
    
    def _check_audio_availability(self):
        """Check if audio file and libraries are available"""
        if not self.enabled:
            self.logger.info("Alert system disabled in config")
            return
        
        if not os.path.exists(self.audio_file):
            self.logger.warning(f"Alert audio file not found: {self.audio_file}")
            self.enabled = False
            return
        
        if not PYAUDIO_AVAILABLE and not PLAYSOUND_AVAILABLE:
            self.logger.warning(
                "Neither pyaudio nor playsound available. "
                "Install with: pip install pyaudio playsound"
            )
            self.enabled = False
            return
        
        self.logger.info(f"Alert system initialized with: {self.audio_file}")
    
    def trigger_alert(self, violation_data=None, async_play=True):
        """
        Trigger violation alert
        
        Args:
            violation_data: Optional violation details
            async_play: Play audio asynchronously
        """
        if not self.enabled:
            return
        
        current_time = time.time()
        self.last_alert_time = current_time
        
        # Log violation
        if violation_data:
            rider_count = violation_data.get('rider_count', 0)
            self.logger.warning(f"ALERT: {rider_count} riders detected on scooter!")
        
        # Play audio
        if async_play:
            threading.Thread(target=self._play_audio, daemon=True).start()
        else:
            self._play_audio()
    
    def _play_audio(self):
        """Play alert audio file"""
        if self.is_playing:
            return
        
        self.is_playing = True
        
        try:
            if PLAYSOUND_AVAILABLE:
                # Use playsound (simpler but blocks)
                playsound(self.audio_file)
            elif PYAUDIO_AVAILABLE:
                # Use pyaudio (more control)
                self._play_audio_pyaudio()
            else:
                self.logger.warning("No audio playback method available")
        except Exception as e:
            self.logger.error(f"Failed to play alert sound: {e}")
        finally:
            self.is_playing = False
    
    def _play_audio_pyaudio(self):
        """Play audio using PyAudio"""
        try:
            wf = wave.open(self.audio_file, 'rb')
            p = pyaudio.PyAudio()
            
            stream = p.open(
                format=p.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                output=True
            )
            
            # Play audio in chunks
            chunk_size = 1024
            data = wf.readframes(chunk_size)
            
            while data:
                stream.write(data)
                data = wf.readframes(chunk_size)
            
            # Cleanup
            stream.stop_stream()
            stream.close()
            p.terminate()
            wf.close()
            
        except Exception as e:
            self.logger.error(f"PyAudio playback failed: {e}")
    
    def draw_visual_alert(self, frame, violations):
        """
        Draw visual overlay for violations
        
        Args:
            frame: Video frame to draw on
            violations: List of violation detections
        
        Returns:
            Modified frame with visual alerts
        """
        if not self.visual_overlay or not violations:
            return frame
        
        import cv2
        
        for violation in violations:
            scooter = violation.get('scooter', {})
            rider_count = violation.get('rider_count', 0)
            bbox = scooter.get('bbox', [])
            
            if len(bbox) == 4:
                x1, y1, x2, y2 = map(int, bbox)
                
                # Draw thick red rectangle
                cv2.rectangle(frame, (x1, y1), (x2, y2), self.overlay_color, 4)
                
                # Draw warning text
                warning_text = f"VIOLATION: {rider_count} RIDERS!"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                thickness = 2
                
                # Get text size for background
                (text_width, text_height), baseline = cv2.getTextSize(
                    warning_text, font, font_scale, thickness
                )
                
                # Draw text background
                text_x = x1
                text_y = y1 - 10
                if text_y < text_height:
                    text_y = y2 + text_height + 10
                
                cv2.rectangle(
                    frame,
                    (text_x, text_y - text_height - baseline),
                    (text_x + text_width, text_y + baseline),
                    self.overlay_color,
                    -1
                )
                
                # Draw text
                cv2.putText(
                    frame,
                    warning_text,
                    (text_x, text_y),
                    font,
                    font_scale,
                    (255, 255, 255),
                    thickness
                )
        
        return frame
    
    def should_show_visual(self):
        """Check if visual alert should still be shown"""
        if self.last_alert_time == 0:
            return False
        
        elapsed = time.time() - self.last_alert_time
        return elapsed < self.overlay_duration
