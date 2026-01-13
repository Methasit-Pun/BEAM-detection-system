"""
E-Scooter Detector Module
Main detection class that orchestrates all components
"""

import cv2
import logging
import time
from .model_loader import ModelLoader
from .video_handler import VideoHandler
from .violation_checker import ViolationChecker
from .alert_system import AlertSystem
from .utils import setup_logging, draw_detections


class EScooterDetector:
    """Main e-scooter detection system"""
    
    def __init__(self, config):
        """
        Initialize detector with configuration
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.model = ModelLoader(
            model_path=config['model']['path'],
            labels_path=config['model']['labels'],
            use_cuda=config['performance'].get('enable_cuda', True)
        )
        
        self.video_handler = VideoHandler(
            source=config['camera']['source'],
            width=config['camera']['width'],
            height=config['camera']['height'],
            fps=config['camera']['fps']
        )
        
        self.violation_checker = ViolationChecker(
            config['violation']
        )
        
        self.alert_system = AlertSystem(
            config['alert']
        )
        
        # Display settings
        self.show_window = config['display'].get('show_window', True)
        self.show_fps = config['display'].get('show_fps', True)
        self.window_name = config['display'].get('window_name', 'E-Scooter Detection')
        
        # Statistics
        self.frame_count = 0
        self.detection_count = 0
        self.violation_count = 0
        self.start_time = time.time()
        
        self.logger.info("E-Scooter detector initialized")
    
    def run(self):
        """Main detection loop"""
        self.logger.info("Starting detection system...")
        
        try:
            while self.video_handler.is_opened():
                # Read frame
                ret, frame = self.video_handler.read()
                if not ret:
                    self.logger.warning("Failed to read frame")
                    break
                
                self.frame_count += 1
                
                # Preprocess and run inference
                input_tensor = self.model.preprocess(frame)
                outputs = self.model.infer(input_tensor)
                
                # Postprocess outputs
                detections = self.model.postprocess(
                    outputs,
                    confidence_threshold=self.config['model']['confidence_threshold']
                )
                
                self.detection_count += len(detections)
                
                # Check for violations
                violations = self.violation_checker.check_violations(detections)
                
                # Trigger alerts if violations found
                if violations:
                    self.violation_count += len(violations)
                    for violation in violations:
                        self.alert_system.trigger_alert(violation)
                
                # Draw visualization
                if self.show_window:
                    frame = self._draw_frame(frame, detections, violations)
                    cv2.imshow(self.window_name, frame)
                    
                    # Check for quit key
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.logger.info("Quit key pressed")
                        break
        
        except KeyboardInterrupt:
            self.logger.info("Interrupted by user")
        
        except Exception as e:
            self.logger.error(f"Error in detection loop: {e}", exc_info=True)
        
        finally:
            self.cleanup()
    
    def _draw_frame(self, frame, detections, violations):
        """
        Draw detections and violations on frame
        
        Args:
            frame: Input frame
            detections: List of detections
            violations: List of violations
        
        Returns:
            Annotated frame
        """
        # Draw all detections
        frame = draw_detections(
            frame,
            detections,
            show_confidence=self.config['display'].get('show_confidence', True),
            thickness=self.config['display'].get('bbox_thickness', 2)
        )
        
        # Draw violation overlays
        frame = self.alert_system.draw_visual_alert(frame, violations)
        
        # Draw FPS
        if self.show_fps:
            fps = self.video_handler.get_fps()
            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
        
        # Draw statistics
        stats_y = 60
        stats = [
            f"Frames: {self.frame_count}",
            f"Detections: {self.detection_count}",
            f"Violations: {self.violation_count}"
        ]
        
        for i, stat in enumerate(stats):
            cv2.putText(
                frame,
                stat,
                (10, stats_y + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
        
        return frame
    
    def cleanup(self):
        """Cleanup resources"""
        self.logger.info("Cleaning up...")
        
        # Print final statistics
        elapsed_time = time.time() - self.start_time
        avg_fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
        
        self.logger.info(f"Total frames processed: {self.frame_count}")
        self.logger.info(f"Total detections: {self.detection_count}")
        self.logger.info(f"Total violations: {self.violation_count}")
        self.logger.info(f"Average FPS: {avg_fps:.2f}")
        self.logger.info(f"Runtime: {elapsed_time:.2f} seconds")
        
        # Release resources
        self.video_handler.release()
        cv2.destroyAllWindows()
        
        self.logger.info("Cleanup complete")
