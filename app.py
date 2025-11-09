import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os
import time

# Set page configuration
st.set_page_config(
    page_title="AI Text Summarizer",
    page_icon="📝",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .summary-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 15px 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the fine-tuned model and tokenizer"""
    try:
        model_dir = "./t5_summarization_model"
        
        if not os.path.exists(model_dir):
            st.error("Model directory not found!")
            return None, None, None
            
        # Check for essential files
        required_files = ['config.json', 'model.safetensors', 'tokenizer.json']
        missing_files = [f for f in required_files if not os.path.exists(os.path.join(model_dir, f))]
        
        if missing_files:
            st.error(f"Missing files: {', '.join(missing_files)}")
            return None, None, None
        
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        st.success("✅ Model loaded successfully!")
        return model, tokenizer, device
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None

def generate_summary(model, tokenizer, device, text, max_length=100):
    """Generate summary using the fine-tuned model"""
    try:
        # Add T5 prefix
        input_text = "summarize: " + text
        
        # Tokenize
        inputs = tokenizer(
            input_text, 
            max_length=512, 
            truncation=True, 
            padding=True,
            return_tensors="pt"
        )
        
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)
        
        # Generate summary
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                num_beams=2,
                length_penalty=2.0,
                early_stopping=True
            )
        
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return summary
        
    except Exception as e:
        st.error(f"Error generating summary: {str(e)}")
        return None

def main():
    st.title("📝 AI Text Summarizer")
    st.markdown("Transform long articles into concise summaries")
    
    # Sidebar settings
    st.sidebar.title("Settings")
    max_length = st.sidebar.slider("Summary Length", 50, 150, 100)
    
    # Load model
    model, tokenizer, device = load_model()
    
    if model is None:
        st.error("""
        ❌ Model failed to load. Please check:
        - All model files are in 't5_summarization_model' folder
        - Required files: config.json, model.safetensors, tokenizer files
        """)
        return
    
    # Main interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📄 Input Text")
        
        sample_texts = {
            "Technology": "Apple announced new iPhones with better cameras and faster processors. The devices feature USB-C charging and improved battery life.",
            "Science": "NASA's rover collected Martian rock samples that may contain signs of ancient life. The samples will be returned to Earth for analysis.",
            "Environment": "Climate change is causing more extreme weather events worldwide. Scientists urge immediate action to reduce carbon emissions."
        }
        
        input_method = st.radio("Choose input:", ["Type text", "Use sample"])
        
        if input_method == "Type text":
            input_text = st.text_area("Enter text:", height=200)
        else:
            selected = st.selectbox("Sample:", list(sample_texts.keys()))
            input_text = st.text_area("Text:", value=sample_texts[selected], height=200)
    
    with col2:
        st.subheader("📋 Generated Summary")
        
        if st.button("🚀 Generate Summary", type="primary", use_container_width=True):
            if not input_text or len(input_text.strip()) < 20:
                st.warning("Please enter at least 20 characters.")
            else:
                with st.spinner("Generating summary..."):
                    summary = generate_summary(model, tokenizer, device, input_text, max_length)
                
                if summary:
                    st.markdown("### ✅ Summary")
                    st.markdown(f'<div class="summary-box">{summary}</div>', unsafe_allow_html=True)
                    
                    # Statistics
                    orig_words = len(input_text.split())
                    sum_words = len(summary.split())
                    compression = ((orig_words - sum_words) / orig_words) * 100
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Original Words", orig_words)
                        st.metric("Summary Words", sum_words)
                    with col2:
                        st.metric("Reduction", f"{compression:.1f}%")
                    
                    # Download
                    st.download_button(
                        "💾 Download Summary",
                        summary,
                        file_name="summary.txt",
                        mime="text/plain"
                    )
                else:
                    st.error("Failed to generate summary.")
        else:
            st.info("Enter text and click 'Generate Summary'")

if __name__ == "__main__":
    main()
