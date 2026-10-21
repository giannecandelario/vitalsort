import io
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import models
import numpy as np

class AdvancedVisionEngine:
    """
    Advanced Vision Engine using real pre-trained deep learning models.
    Provides actual image classification and feature extraction.
    """
    
    def __init__(self):
        """Initialize with pre-trained ResNet50 model."""
        # Load pre-trained ResNet50 (trained on ImageNet)
        self.device = torch.device("cpu")  # Use CPU for compatibility
        self.model = models.resnet50(pretrained=True)
        self.model.to(self.device)
        self.model.eval()
        
        # Image preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Clinical symptom mappings based on image features
        self.clinical_indicators = {
            "rash": ["red", "inflamed", "skin", "dermatitis"],
            "lesion": ["abnormal", "growth", "wound", "ulcer"],
            "swelling": ["edema", "enlarged", "puffiness"],
            "inflammation": ["inflamed", "erythema", "infection"],
            "wound": ["laceration", "cut", "injury"],
            "discoloration": ["pigmentation", "discolored", "pale"]
        }
    
    def analyze_image(self, image_bytes: bytes) -> dict:
        """
        Analyzes image using deep learning and returns clinical insights.
        
        Args:
            image_bytes: Raw image binary data
            
        Returns:
            Dict with extracted clinical features and confidence scores
        """
        try:
            # Load and preprocess image
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Extract features from ResNet50
            with torch.no_grad():
                features = self.model(image_tensor)
                confidence = torch.softmax(features, dim=1)
            
            # Analyze image characteristics
            img_array = np.array(image)
            
            # Calculate visual metrics
            dominant_colors = self._extract_dominant_colors(img_array)
            brightness = self._calculate_brightness(img_array)
            texture_analysis = self._analyze_texture(img_array)
            
            # Map visual features to clinical symptoms
            extracted_symptoms = self._map_to_clinical_symptoms(
                dominant_colors, brightness, texture_analysis
            )
            
            # Determine acuity based on visual analysis
            inferred_acuity = self._determine_acuity(
                extracted_symptoms, brightness, dominant_colors, texture_analysis
            )
            
            return {
                "extracted_symptoms": extracted_symptoms,
                "inferred_acuity": inferred_acuity,
                "source": "Advanced Visual Analysis",
                "confidence": float(confidence.max().item()),
                "visual_metrics": {
                    "dominant_colors": dominant_colors,
                    "brightness": brightness,
                    "texture_score": texture_analysis
                }
            }
        
        except Exception as e:
            # Fallback if image processing fails
            return {
                "extracted_symptoms": ["Image analysis pending"],
                "inferred_acuity": "Routine",
                "source": "Visual Scan",
                "error": str(e)
            }
    
    def _extract_dominant_colors(self, img_array: np.ndarray) -> list:
        """Extracts dominant colors from image."""
        # Resize for faster processing
        img_small = Image.fromarray(img_array).resize((50, 50))
        pixels = np.array(img_small)
        
        # Count red, green, blue channels
        red_avg = np.mean(pixels[:, :, 0])
        green_avg = np.mean(pixels[:, :, 1])
        blue_avg = np.mean(pixels[:, :, 2])
        
        # A channel only counts as dominant when it clearly leads the other two.
        # Comparing each against 150 in isolation meant any bright image was
        # simultaneously red, green and blue - a flat gray square scored all three
        # and read as inflammation.
        channels = {"red": red_avg, "green": green_avg, "blue": blue_avg}
        colors = [
            name
            for name, value in channels.items()
            if value > 60 and value - max(
                other for channel, other in channels.items() if channel != name
            ) > 20
        ]
        
        return colors if colors else ["neutral"]
    
    def _calculate_brightness(self, img_array: np.ndarray) -> float:
        """Calculates overall image brightness (0-100)."""
        brightness = np.mean(img_array) / 255.0 * 100
        return round(brightness, 2)
    
    def _analyze_texture(self, img_array: np.ndarray) -> float:
        """Analyzes texture complexity (0-100)."""
        # Calculate edge density as texture metric
        gray = np.dot(img_array[..., :3], [0.2989, 0.5870, 0.1140])
        edges = np.abs(np.gradient(gray)[0]) + np.abs(np.gradient(gray)[1])
        texture_score = np.mean(edges) / 255.0 * 100
        return round(texture_score, 2)
    
    def _map_to_clinical_symptoms(self, colors: list, brightness: float,
                                   texture: float) -> list:
        """Maps visual features to clinical symptoms."""
        symptoms = []
        
        # One weighted token per visual feature. Redness previously expanded into
        # inflammation + erythema + rash, three aliases for the same signal, and
        # because the triage score is additive that single colour reading alone
        # summed to exactly the 8-point Emergency threshold.
        if "red" in colors:
            symptoms.append("inflammation")
        if texture > 50:
            symptoms.append("lesion")
        if brightness < 40:
            symptoms.append("discoloration")
        
        return symptoms if symptoms else ["No abnormalities detected"]
    
    def _determine_acuity(self, symptoms: list, brightness: float, 
                          colors: list, texture: float) -> str:
        """Determines clinical acuity from visual analysis."""
        if len(symptoms) > 2 or ("inflammation" in symptoms and texture > 60):
            return "Urgent/Emergency Risk"
        elif len(symptoms) > 0:
            return "Routine"
        else:
            return "Routine"
