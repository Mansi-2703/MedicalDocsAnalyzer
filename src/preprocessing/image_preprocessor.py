"""
Image preprocessing module using OpenCV.
Optimized for OCR on scanned medical documents.
"""

import cv2
import numpy as np
from typing import Tuple, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """Preprocess images for OCR using OpenCV."""
    
    def __init__(self, config: dict = None):
        """
        Initialize preprocessor with configuration.
        
        Args:
            config: Dict with keys like image_dpi, threshold_value, etc.
        """
        self.config = config or {}
        self.dpi = self.config.get("image_dpi", 300)
        self.threshold_value = self.config.get("threshold_value", 127)
        self.morph_kernel = self.config.get("morph_kernel_size", (5, 5))
        self.dilation_iter = self.config.get("dilation_iterations", 2)
        self.bilateral_enabled = self.config.get("bilateral_filter_enabled", True)
    
    def load_image(self, image_path: str) -> np.ndarray:
        """Load image from file."""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        return img
    
    def convert_to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Convert image to grayscale."""
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    def apply_thresholding(self, image: np.ndarray, method: str = "binary") -> np.ndarray:
        """
        Apply thresholding to convert to binary image.
        
        Args:
            image: Grayscale image
            method: "binary", "otsu", or "adaptive"
        """
        if method == "otsu":
            _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif method == "adaptive":
            thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY, 11, 2)
        else:  # binary
            _, thresh = cv2.threshold(image, self.threshold_value, 255, cv2.THRESH_BINARY)
        return thresh
    
    def apply_bilateral_filter(self, image: np.ndarray) -> np.ndarray:
        """Apply bilateral filter to reduce noise while preserving edges."""
        return cv2.bilateralFilter(image, 9, 75, 75)
    
    def apply_morphological_operations(self, image: np.ndarray) -> np.ndarray:
        """Apply dilation to improve text connectivity."""
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, self.morph_kernel)
        return cv2.dilate(image, kernel, iterations=self.dilation_iter)
    
    def deskew_image(self, image: np.ndarray) -> np.ndarray:
        """
        Deskew document image using Hough transform.
        """
        gray = self.convert_to_grayscale(image) if len(image.shape) == 3 else image
        edges = cv2.Canny(gray, 100, 200)
        lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
        
        angle = 0
        if lines is not None:
            angles = []
            for line in lines[:10]:  # Use first 10 lines
                rho, theta = line[0]
                angle_deg = (theta * 180 / np.pi) - 90
                if -45 < angle_deg < 45:
                    angles.append(angle_deg)
            
            if angles:
                angle = np.median(angles)
        
        if angle != 0:
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, rotation_matrix, (w, h),
                                   borderMode=cv2.BORDER_REPLICATE)
        
        return image
    
    def remove_noise(self, image: np.ndarray) -> np.ndarray:
        """Remove noise using morphological operations."""
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        return image
    
    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(image)
    
    def resize_image(self, image: np.ndarray, target_height: int = 2000) -> np.ndarray:
        """Resize image to target height while maintaining aspect ratio."""
        h, w = image.shape[:2]
        scale = target_height / h
        new_width = int(w * scale)
        return cv2.resize(image, (new_width, target_height), interpolation=cv2.INTER_CUBIC)
    
    def preprocess_pipeline(self, image_path: str, aggressive: bool = False) -> np.ndarray:
        """
        Full preprocessing pipeline optimized for medical documents.
        
        Args:
            image_path: Path to input image
            aggressive: If True, apply more aggressive preprocessing
        
        Returns:
            Preprocessed image
        """
        logger.info(f"Preprocessing image: {image_path}")
        
        # Load and initial processing
        image = self.load_image(image_path)
        image = self.deskew_image(image)
        
        # Convert to grayscale
        gray = self.convert_to_grayscale(image)
        
        # Apply bilateral filter if enabled (preserves edges better for medical text)
        if self.bilateral_enabled:
            gray = self.apply_bilateral_filter(gray)
        
        # Enhance contrast
        gray = self.enhance_contrast(gray)
        
        # Apply thresholding (use Otsu for better results on medical docs)
        binary = self.apply_thresholding(gray, method="otsu")
        
        if aggressive:
            # Remove noise
            binary = self.remove_noise(binary)
            # Apply morphological operations
            binary = self.apply_morphological_operations(binary)
        
        # Resize for better OCR (medical docs benefit from higher resolution)
        binary = self.resize_image(binary, target_height=2000)
        
        logger.info("Preprocessing completed successfully")
        return binary
    
    def save_preprocessed(self, image: np.ndarray, output_path: str):
        """Save preprocessed image."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(output_path, image)
        logger.info(f"Saved preprocessed image to {output_path}")


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    config = {
        "image_dpi": 300,
        "threshold_value": 127,
        "morph_kernel_size": (5, 5),
        "dilation_iterations": 2,
        "bilateral_filter_enabled": True,
    }
    
    preprocessor = ImagePreprocessor(config)
    # Example: processed_img = preprocessor.preprocess_pipeline("path/to/image.jpg")
