import streamlit as st
import PyPDF2
import google.generativeai as genai
import os

# --- Configuration ---
# You will need to put your actual API key here or in an environment variable.
API_KEY = st.secrets["GEMINI_API_KEY"] 
genai.configure(api_key=API_KEY)

# --- App UI Setup ---
st.set_page_config(page_title="Literature Review Synthesizer", page_icon="📚")
st.title("📚 Literature Review Synthesizer")
st.write("Upload up to 10 source documents, and AI will synthesize them into a literature review based on your research question.")

# --- User Inputs ---
field_of_study = st.text_input("Field of Study", placeholder="e.g., Psychology, Computer Science")
research_question = st.text_area("Research Question", placeholder="e.g., How does remote work impact employee mental health?")

# --- Citation Selection and Warning ---
citation_styles = [
    "APA 7", 
    "MLA", 
    "Chicago Style footnotes", 
    "Chicago Style endnotes", 
    "Harvard", 
    "IEEE", 
    "AMA", 
    "Bluebook"
]
citation_style = st.selectbox("Select Citation Style", citation_styles)

# Added warning about PDF metadata and citations
st.info("⚠️ **Note:** Always verify the generated citations and bibliography. PDFs often lack clean metadata, so the AI may struggle to find complete author names, publication dates, or journal titles from the raw text.")

uploaded_files = st.file_uploader("Upload Source Documents (PDFs)", type=["pdf"], accept_multiple_files=True)

# --- Helper Function to Extract Text ---
def extract_text_from_pdfs(files):
    combined_text = ""
    for i, file in enumerate(files):
        try:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            combined_text += f"\n\n--- SOURCE {i+1}: {file.name} ---\n{text[:15000]}" # Limiting chars per doc to avoid token limits
        except Exception as e:
            st.error(f"Error reading {file.name}: {e}")
    return combined_text

# --- Main Logic ---
if st.button("Generate Literature Review"):
    if not field_of_study or not research_question:
        st.warning("Please enter both your field of study and research question.")
    elif not uploaded_files:
        st.warning("Please upload at least one document.")
    elif len(uploaded_files) > 10:
        st.warning("Please upload a maximum of 10 documents.")
    else:
        with st.spinner("Reading documents and synthesizing literature... This may take a minute."):
            # 1. Extract text
            source_material = extract_text_from_pdfs(uploaded_files)
            
            # 2. Build the Prompt 
            prompt = f"""
            You are an expert academic writer and researcher in the field of {field_of_study}.
            I have provided the text from several source documents below. 
            
            Based ONLY on these provided sources, write a comprehensive, cohesive literature review 
            answering this research question: "{research_question}"
            
            CRITICAL INSTRUCTIONS FOR CITATIONS AND BIBLIOGRAPHY:
            - Use **{citation_style}** format for all in-text citations.
            - Create a complete bibliography/reference list at the end of the review formatted strictly in **{citation_style}**.
            - Extract author names, publication years, titles, and other relevant metadata from the provided source texts to build these citations. If exact metadata is missing, format the available information (like the file name provided in the text markers) as closely as possible to the {citation_style} guidelines.
            
            Structure the review with:
            - An introduction to the topic and research question.
            - A thematic synthesis grouping the authors' findings and arguments.
            - Identification of any agreements, disagreements, or gaps in the provided texts.
            - A brief conclusion.
            - A formatted Bibliography/Reference list.
            
            SOURCE TEXTS:
            {source_material}
            """
            
            # 3. Call the AI
            try:
                # Using Gemini 1.5 Flash as it is fast and has a massive context window for reading PDFs
                model = genai.GenerativeModel('gemini-1.5-flash')
                response = model.generate_content(prompt)
                
                # 4. Display the Result
                st.subheader("Your Synthesized Literature Review")
                st.markdown(response.text)
                
                # Add a download button
                st.download_button(
                    label="Download as Text File",
                    data=response.text,
                    file_name="literature_review.txt",
                    mime="text/plain"
                )
            except Exception as e:
                st.error(f"An error occurred while communicating with the AI: {e}")
                st.error(f"An error occurred while communicating with the AI: {e}")
