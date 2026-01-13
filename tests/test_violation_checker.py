"""
Test Violation Checker Module
Unit tests for ViolationChecker
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from violation_checker import ViolationChecker


class TestViolationChecker:
    """Test cases for ViolationChecker class"""
    
    @pytest.fixture
    def checker_config(self):
        """Provide test configuration"""
        return {
            'min_riders_for_alert': 2,
            'overlap_threshold': 0.3,
            'cooldown_seconds': 5,
            'confidence_threshold': 0.6
        }
    
    @pytest.fixture
    def violation_checker(self, checker_config):
        """Create ViolationChecker instance"""
        return ViolationChecker(checker_config)
    
    def test_initialization(self, violation_checker):
        """Test checker initialization"""
        assert violation_checker is not None
        assert violation_checker.min_riders == 2
        assert violation_checker.overlap_threshold == 0.3
    
    def test_calculate_iou_overlap(self, violation_checker):
        """Test IoU calculation with overlapping boxes"""
        bbox1 = [100, 100, 200, 200]  # 100x100 box
        bbox2 = [150, 150, 250, 250]  # 100x100 box, 50% overlap
        
        iou = violation_checker._calculate_iou(bbox1, bbox2)
        assert iou > 0
        assert iou < 1
    
    def test_calculate_iou_no_overlap(self, violation_checker):
        """Test IoU calculation with non-overlapping boxes"""
        bbox1 = [0, 0, 100, 100]
        bbox2 = [200, 200, 300, 300]
        
        iou = violation_checker._calculate_iou(bbox1, bbox2)
        assert iou == 0.0
    
    def test_calculate_iou_identical(self, violation_checker):
        """Test IoU calculation with identical boxes"""
        bbox1 = [100, 100, 200, 200]
        bbox2 = [100, 100, 200, 200]
        
        iou = violation_checker._calculate_iou(bbox1, bbox2)
        assert iou == 1.0
    
    def test_no_violation_single_rider(self, violation_checker):
        """Test no violation detected with single rider"""
        detections = [
            {
                'class_name': 'scooter',
                'bbox': [100, 100, 200, 200],
                'confidence': 0.9
            },
            {
                'class_name': 'person',
                'bbox': [110, 110, 190, 190],
                'confidence': 0.85
            }
        ]
        
        violations = violation_checker.check_violations(detections)
        assert len(violations) == 0
    
    def test_violation_multiple_riders(self, violation_checker):
        """Test violation detected with multiple riders"""
        detections = [
            {
                'class_name': 'beam',
                'bbox': [100, 100, 200, 200],
                'confidence': 0.9
            },
            {
                'class_name': 'person',
                'bbox': [110, 110, 150, 180],
                'confidence': 0.85
            },
            {
                'class_name': 'person',
                'bbox': [150, 110, 190, 180],
                'confidence': 0.82
            }
        ]
        
        violations = violation_checker.check_violations(detections)
        assert len(violations) >= 1
        assert violations[0]['rider_count'] >= 2
    
    def test_low_confidence_filtered(self, violation_checker):
        """Test low confidence detections are filtered"""
        detections = [
            {
                'class_name': 'scooter',
                'bbox': [100, 100, 200, 200],
                'confidence': 0.3  # Below threshold
            },
            {
                'class_name': 'person',
                'bbox': [110, 110, 190, 190],
                'confidence': 0.85
            }
        ]
        
        violations = violation_checker.check_violations(detections)
        assert len(violations) == 0
    
    def test_get_statistics(self, violation_checker):
        """Test statistics retrieval"""
        stats = violation_checker.get_statistics()
        assert 'total_violations' in stats
        assert 'active_cooldowns' in stats
        assert stats['total_violations'] == 0
    
    def test_reset_statistics(self, violation_checker):
        """Test statistics reset"""
        violation_checker.violation_count = 10
        violation_checker.reset_statistics()
        
        stats = violation_checker.get_statistics()
        assert stats['total_violations'] == 0


if __name__ == '__main__':
    pytest.main([__file__])
