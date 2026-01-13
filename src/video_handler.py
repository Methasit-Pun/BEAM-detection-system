"""
Video Handler Module
Manages video capture and frame processing
"""

import cv2
import logging
import time
from collections import deque


class VideoHandler:
    """Handle video input from camera or file"""
    
    def __init__(self, source, width=1920, height=1080, fps=30):
        """
        Initialize video handler
        
        Args:
            source: Video source (camera index, CSI path, or file path)
            width: Frame width
            height: Frame height
            fps: Target frames per second
        """
        self.source = source
        self.width = width
        self.height = height
        self.target_fps = fps
        self.cap = None
        self.logger = logging.getLogger(__name__)
        self.fps_calculator = FPSCalculator()
        
        self._open_camera()
    
    def _open_camera(self):
        """Open video capture device"""
        try:
            # Handle CSI camera format
            if isinstance(self.source, str) and self.source.startswith('csi://'):
                camera_id = int(self.source.replace('csi://', ''))
                gstreamer_pipeline = self._get_gstreamer_pipeline(camera_id)
                self.cap = cv2.VideoCapture(gstreamer_pipeline, cv2.CAP_GSTREAMER)
            
            # Handle regular camera or file
            elif isinstance(self.source, str) and self.source.startswith('/dev/video'):
                camera_id = int(self.source.replace('/dev/video', ''))
                self.cap = cv2.VideoCapture(camera_id)
            
            else:
                # Try as camera index or file path
                try:
                    source_int = int(self.source)
                    self.cap = cv2.VideoCapture(source_int)
                except ValueError:
                    self.cap = cv2.VideoCapture(self.source)
            
            if not self.cap.isOpened():
                raise RuntimeError(f"Failed to open video source: {self.source}")
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
            
            # Get actual properties
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            self.logger.info(f"Video source opened: {self.source}")
            self.logger.info(f"Resolution: {actual_width}x{actual_height} @ {actual_fps} FPS")
            
        except Exception as e:
            self.logger.error(f"Failed to open camera: {e}")
            raise
    
    def _get_gstreamer_pipeline(self, camera_id=0):
        """
        Build GStreamer pipeline for CSI camera on Jetson
        
        Args:
            camera_id: Camera device ID
        
        Returns:
            GStreamer pipeline string
        """
        return (
            f"nvarguscamerasrc sensor-id={camera_id} ! "
            f"video/x-raw(memory:NVMM), width=(int){self.width}, height=(int){self.height}, "
            f"format=(string)NV12, framerate=(fraction){self.target_fps}/1 ! "
            f"nvvidconv flip-method=0 ! "
            f"video/x-raw, width=(int){self.width}, height=(int){self.height}, format=(string)BGRx ! "
            f"videoconvert ! "
            f"video/x-raw, format=(string)BGR ! appsink"
        )
    
    def read(self):
        """
        Read a frame from video source
        
        Returns:
            tuple: (success, frame)
        """
        if self.cap is None or not self.cap.isOpened():
            return False, None
        
        ret, frame = self.cap.read()
        
        if ret:
            self.fps_calculator.update()
        
        return ret, frame
    
    def get_fps(self):
        """Get current FPS"""
        return self.fps_calculator.get_fps()
    
    def release(self):
        """Release video capture"""
        if self.cap is not None:
            self.cap.release()
            self.logger.info("Video source released")
    
    def is_opened(self):
        """Check if video source is open"""
        return self.cap is not None and self.cap.isOpened()
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.release()


class FPSCalculator:
    """Calculate frames per second"""
    
    def __init__(self, buffer_size=30):
        """
        Initialize FPS calculator
        
        Args:
            buffer_size: Number of frames to average over
        """
        self.buffer_size = buffer_size
        self.timestamps = deque(maxlen=buffer_size)
        self.start_time = time.time()
    
    def update(self):
        """Update with new frame timestamp"""
        self.timestamps.append(time.time())
    
    def get_fps(self):
        """
        Calculate current FPS
        
        Returns:
            float: Current FPS
        """
        if len(self.timestamps) < 2:
            return 0.0
        
        time_diff = self.timestamps[-1] - self.timestamps[0]
        if time_diff == 0:
            return 0.0
        
        return (len(self.timestamps) - 1) / time_diff
