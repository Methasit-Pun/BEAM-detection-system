"""
Model Loader Module
Handles loading and managing the SSD-MobileNet ONNX model
"""

import os
import logging
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None

try:
    import tensorrt as trt
except ImportError:
    trt = None


class ModelLoader:
    """Load and manage object detection models"""
    
    def __init__(self, model_path, labels_path, use_cuda=True):
        """
        Initialize model loader
        
        Args:
            model_path: Path to ONNX model file
            labels_path: Path to class labels file
            use_cuda: Enable CUDA acceleration
        """
        self.model_path = model_path
        self.labels_path = labels_path
        self.use_cuda = use_cuda
        self.session = None
        self.labels = []
        self.input_name = None
        self.output_names = None
        self.logger = logging.getLogger(__name__)
        
        self._load_model()
        self._load_labels()
    
    def _load_model(self):
        """Load ONNX model with ONNX Runtime"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        if ort is None:
            raise ImportError("onnxruntime not installed. Install with: pip install onnxruntime")
        
        # Configure execution providers
        providers = []
        if self.use_cuda:
            providers.append('CUDAExecutionProvider')
        providers.append('CPUExecutionProvider')
        
        try:
            self.session = ort.InferenceSession(self.model_path, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [output.name for output in self.session.get_outputs()]
            
            self.logger.info(f"Model loaded: {self.model_path}")
            self.logger.info(f"Input: {self.input_name}")
            self.logger.info(f"Outputs: {self.output_names}")
            self.logger.info(f"Providers: {self.session.get_providers()}")
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def _load_labels(self):
        """Load class labels from file"""
        if not os.path.exists(self.labels_path):
            self.logger.warning(f"Labels file not found: {self.labels_path}")
            self.labels = []
            return
        
        try:
            with open(self.labels_path, 'r') as f:
                self.labels = [line.strip() for line in f.readlines()]
            self.logger.info(f"Loaded {len(self.labels)} class labels")
        except Exception as e:
            self.logger.error(f"Failed to load labels: {e}")
            self.labels = []
    
    def preprocess(self, image, input_size=(300, 300)):
        """
        Preprocess image for model input
        
        Args:
            image: Input image (numpy array)
            input_size: Target size (width, height)
        
        Returns:
            Preprocessed image tensor
        """
        import cv2
        
        # Resize
        resized = cv2.resize(image, input_size)
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        normalized = rgb.astype(np.float32) / 255.0
        
        # Add batch dimension and transpose to NCHW format
        # Shape: (1, 3, height, width)
        input_tensor = np.transpose(normalized, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0)
        
        return input_tensor
    
    def infer(self, input_tensor):
        """
        Run inference on input tensor
        
        Args:
            input_tensor: Preprocessed input tensor
        
        Returns:
            Model outputs (boxes, scores, classes)
        """
        try:
            outputs = self.session.run(
                self.output_names,
                {self.input_name: input_tensor}
            )
            return outputs
        except Exception as e:
            self.logger.error(f"Inference failed: {e}")
            return None
    
    def postprocess(self, outputs, confidence_threshold=0.5):
        """
        Process model outputs to extract detections
        
        Args:
            outputs: Raw model outputs
            confidence_threshold: Minimum confidence score
        
        Returns:
            List of detections with bounding boxes, classes, and scores
        """
        detections = []
        
        if outputs is None or len(outputs) < 2:
            return detections
        
        # Assuming outputs format: [boxes, scores, classes]
        boxes = outputs[0]  # Shape: (N, 4)
        scores = outputs[1]  # Shape: (N,)
        
        # Filter by confidence
        for i in range(len(scores)):
            if scores[i] >= confidence_threshold:
                detection = {
                    'bbox': boxes[i].tolist(),
                    'confidence': float(scores[i]),
                    'class_id': i % len(self.labels) if self.labels else 0,
                    'class_name': self.labels[i % len(self.labels)] if self.labels else 'unknown'
                }
                detections.append(detection)
        
        return detections
    
    def get_input_shape(self):
        """Get expected input shape"""
        if self.session:
            return self.session.get_inputs()[0].shape
        return None
