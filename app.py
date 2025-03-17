import streamlit as st
import base64
from groq import Groq
from groq_api_key import groq_api_key  # Import the Groq API key

# Initialize Groq client with the imported key
client = Groq(api_key=groq_api_key)

# System prompt for medical image analysis
system_prompt = """
You are a domain expert in medical image analysis. You are tasked with 
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
which can help in faster recovery.

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

# Streamlit UI configuration
st.set_page_config(page_title="Visual Medical Assistant", page_icon="🩺", layout="wide")
st.title("Visual Medical Assistant 👨‍⚕️ 🩺 🏥")
st.subheader("An app to help with medical analysis using images")

# File uploader
file_uploaded = st.file_uploader("Upload the image for Analysis", type=["png", "jpg", "jpeg"])

# Display uploaded image
if file_uploaded:
    st.image(file_uploaded, width=200, caption="Uploaded Image")

# Submit button
submit = st.button("Generate Analysis")

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
            with st.spinner("Generating analysis..."):
                response = client.chat.completions.create(
                    model="llama-3.2-11b-vision-preview",  # Multimodal model
                    messages=messages,
                    max_tokens=8192,
                    temperature=1.0,
                )

            # Display response
            if response and response.choices:
                st.title("Detailed Analysis Based on the Uploaded Image")
                st.markdown(response.choices[0].message.content)
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
st.write("Powered by Groq API | Date: March 17, 2025")
