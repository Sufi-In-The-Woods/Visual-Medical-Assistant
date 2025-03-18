import streamlit as st
import base64
from groq import Groq
from groq_api_key import groq_api_key  # Import the Groq API key

# Initialize Groq client with the imported key
client = Groq(api_key=groq_api_key)

# Streamlit UI configuration (MUST BE FIRST STREAMLIT COMMAND)
st.set_page_config(page_title="MediScript Analyzer", page_icon="📝", layout="wide")

# Custom CSS for improved UI (after set_page_config)
st.markdown("""
    <style>
    .main-title {
        font-size: 2.5em;
        font-weight: bold;
        color: #1E90FF;
        text-align: center;
        margin-bottom: 0.5em;
    }
    .subheader {
        font-size: 1.5em;
        color: #4682B4;
        text-align: center;
        margin-bottom: 1em;
    }
    .section-header {
        font-size: 1.8em;
        color: #1E90FF;
        margin-top: 1.5em;
        margin-bottom: 0.5em;
    }
    .footer-text {
        font-size: 0.9em;
        color: #666;
        text-align: center;
        margin-top: 2em;
    }
    .stButton>button {
        background-color: #1E90FF;
        color: white;
        border-radius: 5px;
        padding: 0.5em 1em;
    }
    .stButton>button:hover {
        background-color: #4682B4;
    }
    </style>
""", unsafe_allow_html=True)

# System prompt (unchanged)
system_prompt = """
You are an expert in analyzing medical documents, including prescriptions and medical test reports. You are tasked with examining images of these documents (handwritten or typed) to extract and interpret key information for a healthcare provider. Your expertise will help in identifying medication details from prescriptions or explaining test results from medical reports.

Your key responsibilities depend on the document type:

For Prescriptions:
1. Detailed Analysis: Scrutinize the prescription image to identify all visible text, focusing on medication names, dosages, administration instructions, and any additional notes.
2. Extracted Information: Document the findings in a clear, structured format, including:
   - Medication Name(s)
   - Dosage(s)
   - Frequency (e.g., daily, twice daily)
   - Route of Administration (e.g., oral, injection)
   - Duration (if specified)
   - Additional Instructions (if any)
3. Potential Issues: Highlight any unclear text, illegible handwriting, or ambiguous instructions that may require clarification.

For Medical Test Reports:
1. Detailed Analysis: Examine the test report image to identify all visible text, focusing on test names, results, reference ranges, and any comments or interpretations.
2. Extracted Information: Document the findings in a clear, structured format, including:
   - Test Name(s)
   - Result(s) (with units, if applicable)
   - Reference Range(s)
   - Abnormal Findings (if any, e.g., values outside reference ranges)
   - Comments or Notes (if present)
3. Potential Issues: Highlight any unclear text, missing data, or ambiguous results that may require further investigation.

General Responsibilities:
4. Explanation: Provide a brief explanation of the extracted information (e.g., what a medication is used for, or what a test result might indicate), tailored to the document type.
5. Disclaimer: Include the disclaimer: "Verify with a pharmacist or doctor before acting on this analysis. This is not a substitute for professional medical advice."

Important Notes:
1. Scope: Only analyze images of prescriptions or medical test reports related to human health.
2. Clarity: If the image is unclear, note that certain details are 'Unable to be accurately determined based on the uploaded image'.
3. Detection: Automatically determine whether the image is a prescription or a medical test report based on its content (e.g., presence of medication names vs. test results). If uncertain, analyze it as both and note the ambiguity.
4. Your analysis assists healthcare professionals but is not a definitive diagnosis or prescription.

Please provide the final response with these headings:
- Detailed Analysis
- Extracted Information
- Potential Issues
- Explanation
- Disclaimer
"""

# Logo at the top
st.image("https://i.postimg.cc/ZRSsW8hC/logo.png", width=300, use_column_width="auto", caption=None)

# Title and subheader
st.markdown('<div class="main-title">MediScript Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader">Analyze prescriptions and medical test reports with AI precision</div>', unsafe_allow_html=True)

# Main content in a container
with st.container():
    col1, col2 = st.columns([1, 2])  # Two-column layout
    
    with col1:
        # File uploader
        file_uploaded = st.file_uploader("Upload a prescription or medical test report", type=["png", "jpg", "jpeg"], help="Supported formats: PNG, JPG, JPEG")
        
        # Display uploaded image
        if file_uploaded:
            st.image(file_uploaded, width=200, caption="Uploaded Document")
        
        # Submit button
        submit = st.button("Analyze Document")

    with col2:
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
                    with st.spinner("Analyzing document..."):
                        response = client.chat.completions.create(
                            model="llama-3.2-11b-vision-preview",  # Multimodal model
                            messages=messages,
                            max_tokens=8192,
                            temperature=1.0,
                        )

                    # Display response
                    if response and response.choices:
                        st.markdown('<div class="section-header">Analysis Results</div>', unsafe_allow_html=True)
                        st.markdown(response.choices[0].message.content)
                    else:
                        st.error("No response received from the API.")

            except Exception as e:
                st.error(f"Error analyzing document: {str(e)}")
                if isinstance(e, client.APIConnectionError):
                    st.write("Could not connect to the Groq API. Check your network or API key.")
                elif isinstance(e, client.RateLimitError):
                    st.write("Rate limit exceeded. Please wait and try again.")
                elif isinstance(e, client.APIStatusError):
                    st.write(f"API returned an error: {e.status_code} - {e.response}")

# How It Works Section
st.markdown('<div class="section-header">How MediScript Analyzer Works</div>', unsafe_allow_html=True)
st.markdown("""
MediScript Analyzer leverages cutting-edge AI technology powered by Groq’s multimodal models to process images of prescriptions and medical test reports. Here’s how it works:

- **Image Processing**: Upload an image, and our system converts it into a format readable by AI.
- **Text Extraction**: The model identifies and extracts key text, whether it’s medication details from a prescription or lab results from a test report.
- **Intelligent Analysis**: Using advanced vision and language understanding, it interprets the data, detects potential issues, and provides explanations (e.g., what a medication treats or what a test result suggests).
- **Structured Output**: Results are presented clearly under headings like Detailed Analysis, Extracted Information, and more, making it easy for healthcare providers or users to understand.

**EarlyMed’s Vision**: Developed by our team at VIT-AP University, EarlyMed aims to empower users to stay proactive about their health. By integrating AI tools like MediScript Analyzer, we’re building a platform to simplify medical document interpretation, promote early detection of health issues, and bridge the gap between technology and well-being. Our goal is a healthier, more informed community.
""")

# Footer
st.markdown("---")
st.markdown("""
<div class="footer-text">
A side project of EarlyMed, a platform developed by our team at VIT-AP University.<br>
Our goal is to help users stay aware of their health and leverage technology and AI for a healthier life.
</div>
""", unsafe_allow_html=True)
st.write("Developed by Team Sysiphus of EarlyMed")
