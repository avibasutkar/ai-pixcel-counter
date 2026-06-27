import os
import io
import unittest
import numpy as np
from PIL import Image
from config import Config

# Try importing the pipeline
try:
    from models.segmentation_model import SegmentationModel
    from utils.image_loader import ImageLoader
    from utils.preprocessor import Preprocessor
    from utils.pixel_counter import PixelCounter
    from utils.visualizer import Visualizer
except ImportError as e:
    raise RuntimeError(f"Missing core module. Are all files present? {e}")

class TestValidation(unittest.TestCase):
    
    def setUp(self):
        # Create a tiny dummy image for tests
        self.dummy_img = Image.new('RGB', (100, 100), color = 'red')
        
    def test_imports(self):
        """Test 1: Check if all necessary modules are properly imported."""
        self.assertTrue(True, "Imports successful if this code executes")
        
    def test_config_validation(self):
        """Test 2: Verify all config parameters."""
        self.assertTrue(Config.check_keys(), "Config validation failed")
        self.assertIn("DeepLabV3", Config.MODELS_AVAILABLE)
        
    def test_model_inference(self):
        """Test 3: Test model loads and correctly returns a mask shape without crashing."""
        model = SegmentationModel()
        mask = model.predict(self.dummy_img)
        self.assertEqual(mask.shape, (100, 100), "Mask shape should match input image size")
        
    def test_pixel_counting(self):
        """Test 4: Verify that pixels are counted and separated properly."""
        # Create a fake mask: 50x100 of class 1 (person), 50x100 of class 0 (background)
        fake_mask = np.zeros((100, 100), dtype=np.uint8)
        fake_mask[50:, :] = 1 
        
        counts = PixelCounter.count_pixels(fake_mask)
        self.assertIn('__background__', counts)
        self.assertIn('person', counts)
        self.assertEqual(counts['__background__'], 5000)
        self.assertEqual(counts['person'], 5000)
        
    def test_end_to_end_pipeline(self):
        """Test 5: E2E test going through resize, infer, mask visual without breaks."""
        large_img = Image.new('RGB', (2000, 2000), color='blue')
        proc_img = Preprocessor.resize_safely(large_img)
        self.assertLessEqual(max(proc_img.size), Preprocessor.MAX_DIMENSION)
        
        model = SegmentationModel()
        mask = model.predict(proc_img)
        
        overlay = Visualizer.overlay_mask(proc_img, mask)
        self.assertEqual(overlay.size, proc_img.size, "Overlay should match processed image size")

if __name__ == '__main__':
    print("Running 5 Strict Validation Tests:")
    unittest.main(verbosity=2)
