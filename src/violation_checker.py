"""
Violation Checker Module
Detects multi-rider violations on e-scooters
"""

import logging
import time
from collections import defaultdict


class ViolationChecker:
    """Check for e-scooter violation rules"""
    
    def __init__(self, config):
        """
        Initialize violation checker
        
        Args:
            config: Configuration dictionary with violation settings
        """
        self.config = config
        self.min_riders = config.get('min_riders_for_alert', 2)
        self.overlap_threshold = config.get('overlap_threshold', 0.3)
        self.cooldown_seconds = config.get('cooldown_seconds', 5)
        self.confidence_threshold = config.get('confidence_threshold', 0.6)
        
        self.last_alert_time = defaultdict(float)
        self.violation_count = 0
        self.logger = logging.getLogger(__name__)
    
    def check_violations(self, detections):
        """
        Check for violations in detections
        
        Args:
            detections: List of detection dictionaries
        
        Returns:
            list: Violations detected (scooter bbox, rider count, confidence)
        """
        # Separate scooters and people
        scooters = []
        people = []
        
        for det in detections:
            class_name = det.get('class_name', '').lower()
            confidence = det.get('confidence', 0.0)
            bbox = det.get('bbox', [])
            
            if confidence < self.confidence_threshold:
                continue
            
            if 'scooter' in class_name or 'beam' in class_name:
                scooters.append(det)
            elif 'person' in class_name:
                people.append(det)
        
        # Check each scooter for multiple riders
        violations = []
        current_time = time.time()
        
        for scooter in scooters:
            scooter_bbox = scooter['bbox']
            rider_count = 0
            overlapping_people = []
            
            # Count people overlapping with scooter
            for person in people:
                person_bbox = person['bbox']
                iou = self._calculate_iou(scooter_bbox, person_bbox)
                
                if iou >= self.overlap_threshold:
                    rider_count += 1
                    overlapping_people.append(person)
            
            # Check if violation (multiple riders)
            if rider_count >= self.min_riders:
                scooter_id = self._get_scooter_id(scooter_bbox)
                
                # Check cooldown to avoid repeated alerts
                if current_time - self.last_alert_time[scooter_id] >= self.cooldown_seconds:
                    violation = {
                        'scooter': scooter,
                        'rider_count': rider_count,
                        'people': overlapping_people,
                        'timestamp': current_time
                    }
                    violations.append(violation)
                    
                    self.last_alert_time[scooter_id] = current_time
                    self.violation_count += 1
                    
                    self.logger.warning(
                        f"Violation detected: {rider_count} riders on scooter "
                        f"(confidence: {scooter['confidence']:.2f})"
                    )
        
        return violations
    
    def _calculate_iou(self, bbox1, bbox2):
        """
        Calculate Intersection over Union (IoU) between two bounding boxes
        
        Args:
            bbox1: [x1, y1, x2, y2] or [x, y, w, h]
            bbox2: [x1, y1, x2, y2] or [x, y, w, h]
        
        Returns:
            float: IoU value
        """
        # Convert to [x1, y1, x2, y2] format if needed
        if len(bbox1) == 4 and len(bbox2) == 4:
            # Assume already in [x1, y1, x2, y2] format
            x1_1, y1_1, x2_1, y2_1 = bbox1
            x1_2, y1_2, x2_2, y2_2 = bbox2
        else:
            return 0.0
        
        # Calculate intersection area
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union area
        bbox1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        bbox2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = bbox1_area + bbox2_area - intersection_area
        
        if union_area == 0:
            return 0.0
        
        iou = intersection_area / union_area
        return iou
    
    def _get_scooter_id(self, bbox):
        """
        Generate simple ID for scooter based on position
        
        Args:
            bbox: Bounding box [x1, y1, x2, y2]
        
        Returns:
            str: Scooter ID
        """
        # Use center position rounded to grid
        center_x = int((bbox[0] + bbox[2]) / 2 / 100) * 100
        center_y = int((bbox[1] + bbox[3]) / 2 / 100) * 100
        return f"scooter_{center_x}_{center_y}"
    
    def get_statistics(self):
        """
        Get violation statistics
        
        Returns:
            dict: Statistics
        """
        return {
            'total_violations': self.violation_count,
            'active_cooldowns': len(self.last_alert_time)
        }
    
    def reset_statistics(self):
        """Reset violation counters"""
        self.violation_count = 0
        self.last_alert_time.clear()
