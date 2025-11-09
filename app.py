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

st.title("📝 AI Text Summarizer (Fine-tuned)")
st.write("Using your custom fine-tuned model for summarization")

@st.cache_resource
def load_model_optimized():
    """Load your fine-tuned model with memory optimization"""
    try:
        st.info("🔄 Loading your fine-tuned model...")
        
        # Memory optimization settings
        tokenizer = AutoTokenizer.from_pretrained(".")
        
        # Load model with optimizations
        model = AutoModelForSeq2SeqLM.from_pretrained(
            ".",
            torch_dtype=torch.float16,  # Use half precision to save memory
            low_cpu_mem_usage=True,
            device_map="auto"
        )
        
        st.success("✅ Your fine-tuned model loaded successfully!")
        return model, tokenizer
        
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        
        # Fallback: Try without optimizations
        try:
            st.info("🔄 Trying alternative loading method...")
            model = AutoModelForSeq2SeqLM.from_pretrained(".")
            tokenizer = AutoTokenizer.from_pretrained(".")
            st.success("✅ Model loaded with fallback method!")
            return model, tokenizer
        except Exception as e2:
            st.error(f"❌ Fallback also failed: {str(e2)}")
            return None, None

def generate_summary_optimized(model, tokenizer, text, max_length=100):
    """Generate summary with memory optimization"""
    try:
        # Use CPU for inference to save GPU memory
        device = torch.device("cpu")
        model = model.to(device)
        
        input_text = "summarize: " + text
        
        inputs = tokenizer(
            input_text, 
            max_length=512, 
            truncation=True, 
            padding=True,
            return_tensors="pt"
        )
        
        # Move to CPU
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                num_beams=2,  # Reduced from 4 to save memory
                early_stopping=True,
                no_repeat_ngram_size=2
            )
        
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return summary
        
    except Exception as e:
        st.error(f"Error generating summary: {str(e)}")
        return None

# Load the model
model, tokenizer = load_model_optimized()

if model is None:
    st.error("""
    **Memory Issue Detected!**
    
    Your fine-tuned model is too large for Streamlit Cloud's memory limits.
    
    **Next Steps:**
    1. Try the solutions below
    2. Consider using a different deployment platform
    """)
    
    st.info("**Alternative Deployment Platforms:**")
    st.write("• **Hugging Face Spaces** (Recommended - better memory limits)")
    st.write("• **Railway.app**")
    st.write("• **Render.com**")
    st.write("• **Google Cloud Run**")

else:
    # Main application
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📄 Input Text")
        text = st.text_area("Enter article:", height=300,
                          placeholder="Paste your article here...")
    
    with col2:
        st.subheader("📋 Summary")
        
        if st.button("🚀 Generate Summary", type="primary"):
            if text:
                with st.spinner("Generating summary with your fine-tuned model..."):
                    summary = generate_summary_optimized(model, tokenizer, text)
                
                if summary:
                    st.success("✅ Summary generated using your fine-tuned model!")
                    st.text_area("Summary:", value=summary, height=150)
                    
                    # Stats
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Input Words", len(text.split()))
                    with col2:
                        st.metric("Summary Words", len(summary.split()))
            else:
                st.warning("Please enter some text")
