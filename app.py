import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os

# Set page configuration
st.set_page_config(
    page_title="AI Text Summarizer",
    page_icon="📝",
    layout="wide"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .summary-box {
        background-color: #f0f2f6;
        padding: 25px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 15px 0;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def load_model():
    """Load YOUR fine-tuned model"""
    try:
        # Show loading message
        st.info("🔄 Loading YOUR fine-tuned model... This may take a minute.")
        
        # Load tokenizer first
        tokenizer = AutoTokenizer.from_pretrained(".")
        
        # Load model with memory optimizations
        model = AutoModelForSeq2SeqLM.from_pretrained(
            ".",
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        
        # Use CPU for inference (more stable on Streamlit)
        device = torch.device("cpu")
        model = model.to(device)
        model.eval()
        
        st.success("✅ YOUR fine-tuned model loaded successfully!")
        return model, tokenizer, device
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        return None, None, None

def generate_summary(model, tokenizer, device, text, max_length=128):
    """Generate summary using YOUR model"""
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
        
        # Generate with your model
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                num_beams=4,
                length_penalty=2.0,
                early_stopping=True,
                no_repeat_ngram_size=3
            )
        
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return summary
        
    except Exception as e:
        st.error(f"Generation error: {str(e)}")
        return None

def main():
    # Header
    st.markdown('<h1 class="main-header">📝 AI Text Summarizer</h1>', unsafe_allow_html=True)
    st.markdown("### Using YOUR Fine-tuned Model")
    
    # Initialize model
    if 'model_loaded' not in st.session_state:
        model, tokenizer, device = load_model()
        if model is not None:
            st.session_state.model = model
            st.session_state.tokenizer = tokenizer
            st.session_state.device = device
            st.session_state.model_loaded = True
        else:
            st.session_state.model_loaded = False
    
    # Sidebar
    st.sidebar.title("⚙️ Settings")
    max_length = st.sidebar.slider("Summary Length", 50, 200, 128)
    
    # Show file check
    with st.sidebar:
        st.markdown("### 📁 Model Files Check")
        required_files = ['config.json', 'model.safetensors', 'tokenizer.json']
        for file in required_files:
            if os.path.exists(file):
                st.success(f"✅ {file}")
            else:
                st.error(f"❌ {file}")
    
    if not st.session_state.model_loaded:
        st.error("""
        **Model failed to load. Please check:**
        1. All model files are in the root directory
        2. Files include: config.json, model.safetensors, tokenizer.json, etc.
        3. Wait a few minutes and reload the app
        """)
        return
    
    # Main interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📄 Input Text")
        
        sample_texts = {
            "Technology": """
            Apple has announced the launch of its new iPhone 15 series, featuring improved cameras, 
            a faster A17 processor, and longer battery life. The new models also include USB-C charging, 
            a first for iPhones, and come in a variety of colors. Pre-orders begin this week with 
            shipping expected next month. Analysts predict strong sales due to high consumer demand 
            and the popularity of the upgraded camera system.
            """,
            "Science": """
            NASA's Perseverance rover has successfully collected its first samples of Martian rock, 
            marking a historic step in humanity's exploration of Mars. The samples will be returned 
            to Earth by a future mission, allowing scientists to study the planet's geology and 
            look for signs of ancient life. The rover has been exploring the Jezero Crater, 
            an ancient river delta, since its landing in February 2021.
            """,
            "Environment": """
            The United Nations has issued a warning about the increasing severity of climate change 
            effects worldwide. Rising global temperatures are causing more frequent and intense 
            wildfires, floods, and hurricanes. Governments are urged to take urgent action 
            to reduce carbon emissions and invest in sustainable energy solutions to prevent 
            catastrophic environmental consequences.
            """
        }
        
        input_method = st.radio("Input method:", ["Type text", "Use sample"], horizontal=True)
        
        if input_method == "Type text":
            input_text = st.text_area(
                "Enter your text:",
                height=300,
                placeholder="Paste your article or text here..."
            )
        else:
            selected_sample = st.selectbox("Choose sample:", list(sample_texts.keys()))
            input_text = st.text_area(
                "Sample text:",
                value=sample_texts[selected_sample],
                height=300
            )
    
    with col2:
        st.subheader("📋 Generated Summary")
        
        if st.button("🚀 Generate Summary", type="primary", use_container_width=True):
            if not input_text or len(input_text.strip()) < 30:
                st.warning("Please enter at least 30 characters.")
            else:
                with st.spinner("🔄 Generating summary with YOUR model..."):
                    summary = generate_summary(
                        st.session_state.model,
                        st.session_state.tokenizer, 
                        st.session_state.device,
                        input_text, 
                        max_length
                    )
                
                if summary:
                    st.markdown("### ✅ Summary")
                    st.markdown(f'<div class="summary-box">{summary}</div>', unsafe_allow_html=True)
                    
                    # Statistics
                    orig_words = len(input_text.split())
                    sum_words = len(summary.split())
                    compression = ((orig_words - sum_words) / orig_words) * 100
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Original Words", orig_words)
                    with col2:
                        st.metric("Summary Words", sum_words)
                    with col3:
                        st.metric("Reduction", f"{compression:.1f}%")
                    
                    # Download button
                    st.download_button(
                        "💾 Download Summary",
                        summary,
                        file_name="summary.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
                else:
                    st.error("Failed to generate summary. Please try again.")
        else:
            st.info("""
            **Ready to use YOUR model!**
            
            • Enter text in the left panel
            • Click 'Generate Summary'
            • Get results from your fine-tuned model
            """)

if __name__ == "__main__":
    main()
