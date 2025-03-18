import streamlit as st
import base64
from groq import Groq
from groq_api_key import groq_api_key  

# Initialize Groq client with the imported key
client = Groq(api_key=groq_api_key)

# System prompt for medical image analysis (unchanged)
system_prompt = """
You are a domain expert in medical image analysis. Your name is DiagnoVision AI. You are tasked with 
examining medical images for a renowned hospital.
Your expertise will help in identifying or 
discovering any anomalies, diseases, conditions or
any health issues that might be present in the image.

Your key responsibilities:
1. Detailed Analysis: Scrutinize and thoroughly examine each image, 
focusing on finding any abnormalities.
2. Analysis Report: Document all the findings and 
clearly articulate them in a structured format.
3. Recommendations: Based on the analysis, suggest remedies, 
tests or treatments as applicable.
4. Treatments: If applicable, lay out detailed treatments 
which can help in faster recovery. But also, mention that it's
for concerns. Always suggest a professional.

Important Notes to remember:
1. Scope of response: Only respond if the image pertains to 
human health issues.
2. Clarity of image: In case the image is unclear, 
note that certain aspects are 
'Unable to be correctly determined based on the uploaded image'.
3. Disclaimer: Accompany your analysis with the disclaimer: 
"Consult with a Doctor before making any decisions."
4. Your insights are invaluable in guiding clinical decisions. 
Please proceed with the analysis, adhering to the 
structured approach outlined above.

Please provide the final response with these 4 headings: 
Detailed Analysis, Analysis Report, Recommendations, and Treatments
"""

# Custom CSS for a glassy, professional medical UI
st.markdown("""
    <style>
    /* Background and global styling */
    .stApp {
        background: linear-gradient(135deg, #e6f0fa 0%, #f5f7fa 100%);
        font-family: 'Segoe UI', sans-serif;
    }
    /* Glassy container */
    .glass-container {
        background: rgba(255, 255, 255, 0.8);
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.3);
        margin-bottom: 20px;
    }
    /* Title styling */
    h1 {
        color: #1e3a8a;
        font-weight: 700;
        text-align: center;
    }
    /* Subheader styling */
    h2 {
        color: #3b82f6;
        font-weight: 500;
        text-align: center;
    }
    /* Button styling */
    .stButton>button {
        background-color: #1e3a8a;
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #3b82f6;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.4);
    }
    /* File uploader styling */
    .stFileUploader {
        border: 2px dashed #1e3a8a;
        border-radius: 10px;
        padding: 10px;
        background: rgba(255, 255, 255, 0.9);
    }
    /* Footer styling */
    .footer {
        text-align: center;
        color: #4b5e7e;
        font-size: 14px;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Streamlit UI configuration
st.set_page_config(page_title="DiagnoVision AI", page_icon="🩺", layout="wide")

# Main content in a glassy container
with st.container():
    st.markdown('<div class="glass-container">', unsafe_allow_html=True)
    st.title("DiagnoVision AI 👨‍⚕️ 🩺 🏥")
    st.subheader("AI-Powered Medical Image Analysis")
    st.markdown('</div>', unsafe_allow_html=True)

# File uploader and image display in a glassy container
with st.container():
    st.markdown('<div class="glass-container">', unsafe_allow_html=True)
    file_uploaded = st.file_uploader("Upload a Medical Image for Analysis", type=["png", "jpg", "jpeg"])
    if file_uploaded:
        st.image(file_uploaded, width=300, caption="Uploaded Medical Image")
    submit = st.button("Generate Analysis")
    st.markdown('</div>', unsafe_allow_html=True)

# Process the image and generate analysis
if submit and file_uploaded:
    try:
        # Read image data
        image_data = file_uploaded.getvalue()
        if not image_data:
            st.error("Error: No image data found.")
        else:
            # Determine MIME type dynamically
            mime_type = "image/jpeg" if file_uploaded.type in ["image/jpeg", "image/jpg"] else "image/png"
            
            # Encode image to base64
            base64_image = base64.b64encode(image_data).decode("utf-8")

            # Prepare the prompt with image and text
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": system_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}}
                    ]
                }
            ]

            # Call Groq API
            with st.spinner("Analyzing Image..."):
                response = client.chat.completions.create(
                    model="llama-3.2-11b-vision-preview",  # Multimodal model
                    messages=messages,
                    max_tokens=8192,
                    temperature=1.0,
                )

            # Display response in a glassy container
            if response and response.choices:
                with st.container():
                    st.markdown('<div class="glass-container">', unsafe_allow_html=True)
                    st.title("Analysis Results")
                    st.markdown(response.choices[0].message.content)
                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.error("No response received from the API.")

    except Exception as e:
        st.error(f"Error generating analysis: {str(e)}")
        if isinstance(e, client.APIConnectionError):
            st.write("Could not connect to the Groq API. Check your network or API key.")
        elif isinstance(e, client.RateLimitError):
            st.write("Rate limit exceeded. Please wait and try again.")
        elif isinstance(e, client.APIStatusError):
            st.write(f"API returned an error: {e.status_code} - {e.response}")

# Footer
st.markdown("---")
st.markdown("""
    <div class="footer">
        Developed by Team EarlyMed<br>
        A side project of <b>EarlyMed</b>, a platform developed by our team at <b>VIT-AP University</b>.<br>
        Our goal is to help users stay aware of their health and leverage technology and AI for a healthier life.
    </div>
""", unsafe_allow_html=True)
