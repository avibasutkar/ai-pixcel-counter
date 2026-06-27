import streamlit as st
import json
import io
import pandas as pd
from PIL import Image

from config import Config
from models.segmentation_model import SegmentationModel
from utils.image_loader import ImageLoader
from utils.preprocessor import Preprocessor
from utils.pixel_counter import PixelCounter
from utils.visualizer import Visualizer
from utils.validator import Validator
from utils.logger import get_logger

logger = get_logger(__name__)

st.set_page_config(page_title="AI Pixel Counter", layout="wide")

# Cached model loading for efficient inference requests
@st.cache_resource
def load_model():
    """Loads and caches the segmentation model to prevent reloading per request."""
    return SegmentationModel()

def main():
    st.title("🧩 AI-Based Object Pixel Counter using Semantic Segmentation")
    st.markdown("Upload an image to segment objects and precisely count their pixel footprint.")

    # Sidebar
    st.sidebar.header("Configuration")
    model_choice = st.sidebar.selectbox("Select Model", Config.MODELS_AVAILABLE)
    confidence_thresh = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, Config.MODEL_CONFIDENCE_THRESHOLD)
    
    st.sidebar.markdown(f"**Max Size:** {Config.MAX_IMAGE_SIZE_MB}MB")
    st.sidebar.markdown("**Device Override:** Auto (GPU preferred)")

    # Main UI
    uploaded_file = st.file_uploader("Upload Image (JPEG/PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            # 1. Validation
            Validator.validate_upload(uploaded_file)
            
            # 2. Loading
            image_bytes = uploaded_file.read()
            original_image = ImageLoader.load_from_bytes(image_bytes)
            
            st.subheader("Analysis")
            col1, col2 = st.columns(2)
            with col1:
                st.image(original_image, caption="Original Input", use_container_width=True)
                
            if st.button("Analyze Image", type="primary"):
                with st.spinner("Processing through Deep Neural Network..."):
                    
                    try:
                        # 3. Preprocessing
                        proc_image = Preprocessor.resize_safely(original_image)
                        
                        # 4. Model Inference
                        model = load_model()
                        mask = model.predict(proc_image)
                        
                        # 5. Result Generation
                        overlay = Visualizer.overlay_mask(proc_image, mask)
                        pixel_counts = PixelCounter.count_pixels(mask)
                        
                        raw_w, raw_h = proc_image.size
                        total_pixels = raw_w * raw_h
                        
                        with col2:
                            st.image(overlay, caption="Segmentation Overlay", use_container_width=True)
                            
                        # 6. Result Metrics
                        st.subheader("📊 Extraction Results")
                        st.metric(label="Total Analyzed Pixels", value=total_pixels)
                        
                        if pixel_counts:
                            df = pd.DataFrame(list(pixel_counts.items()), columns=["Class", "Pixel Count"])
                            df["Percentage (%)"] = (df["Pixel Count"] / total_pixels * 100).round(2)
                            
                            st.dataframe(df, use_container_width=True, hide_index=True)
                            
                            # Prepare Download Assets
                            result_json = {"total_pixels": total_pixels, "classes": pixel_counts}
                            
                            overlay_bytes_io = io.BytesIO()
                            overlay.save(overlay_bytes_io, format='PNG')
                            overlay_bytes = overlay_bytes_io.getvalue()
                            
                            dl_col1, dl_col2 = st.columns(2)
                            with dl_col1:
                                st.download_button(
                                    label="📥 Download JSON Results",
                                    data=json.dumps(result_json, indent=2),
                                    file_name="segmentation_results.json",
                                    mime="application/json"
                                )
                            with dl_col2:
                                st.download_button(
                                    label="🖼️ Download Mask Overlay",
                                    data=overlay_bytes,
                                    file_name="mask_overlay.png",
                                    mime="image/png"
                                )
                        else:
                            st.info("No recognizable objects found in the image based on COCO classes.")

                    except Exception as ai_err:
                        logger.error(f"Analysis Error: {ai_err}")
                        st.error(f"Error during AI processing: {ai_err}")
                        
        except ValueError as e:
            st.warning(str(e))
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
