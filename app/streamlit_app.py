import time
import streamlit as st
from PIL import Image
import google.generativeai as genai
import asyncio
from langchain_core.output_parsers import JsonOutputParser


prompt_template_extract_ocr = """
You are tasked with extracting all textual information from the provided image accurately, even if the image is blurry or contains glare. Extract both plain text and structured tabular information.

### Guidelines for Extraction:

1. **General Text Extraction**:
   - Extract all visible text, including titles, headers, annotations, and any other written content.
   - Preserve:
     - The original case (upper/lowercase letters).
     - All punctuation marks and formatting.
   - Ensure all extracted text is direct and accurate—no interpretations or omissions.

2. **Table and Structured Content Extraction**:
   - Identify structured tabular data, prescriptions, or organized lists within the image.
   - Represent these in Markdown format using `|` for columns and `-` for headers.
   - Maintain the integrity of the content: include all rows, columns, and line breaks using `<br>` tags if needed.

3. **Content Organization**:
   - Deliver extracted text in Markdown format.
   - Maintain a logical flow, similar to how the content is presented in the image.
   - Do not interpret or restructure the text beyond what is explicitly visible.

### Specific Requirements:
- Extract the following details in their entirety if visible:
  - Patient Information, including name, age, gender, date, height, weight, BMI, SPO2, HR, B.P, TEM.
  - Hospital Information, including hospital name, location, doctor details (name, specialization, license number, and contact).
  - Medication Information, including medicine name, dosage, form (e.g., tablet, liquid), and quantity (e.g., sheets, bottles).
  - Doctor's notes or any other remarks.

### Output Format:
- Present all extracted data in Markdown format without rephrasing.
- Ensure completeness and high fidelity of the output by avoiding omission or assumption.
- Avoid unnecessary annotations or commentary, providing the content in a raw form.

Your primary goal is to accurately represent all the textual content from the image in a clear and structured Markdown format.

"""

prompt_template_extract_structured_data = """
I have the following raw OCR-extracted text from a medical prescription. Your task is to transform this raw text into a structured JSON format as specified below.

**Requirements:**

1. **Patient Information**:
   - Extract the following details:
     - **Name**: The patient's full name.
     - **Age**: Patient's age.
     - **Gender**: Patient's gender.
     - **Date**: Date of the prescription.
     - **Height, Weight, BMI, SPO2, HR, B.P, TEM**: Extract these metrics if available.

2. **Doctor Information**:
   - **Name**: The name of the doctor.
   - **Graduation**: Doctor's qualifications (e.g., MD, DM Cardiology).
   - **Hospital Name**: The name of the hospital or clinic.
   - **Location**: Address or contact information of the hospital.
   - **License Number**: The doctor's license/registration number.

3. **Medication Information**:
   - Extract the list of medicines prescribed. For each medicine, include:
     - **Medicine Name**: The name of the medicine.
     - **Dosage**: The prescribed dosage (e.g., 100 mg, 5 ml).
     - **Form**: Type of medicine (e.g., tablet, liquid).
     - **Quantity**: Amount prescribed (e.g., sheets, bottles).

4. **Symptoms**:
   - Extract the list of symptoms or reasons for prescription mentioned by the doctor (e.g., hypothyroidism, neuropathy).

5. **Doctor's Note**:
   - Include any additional remarks or notes provided by the doctor.

**Output Format**:
```json
{
  "Patient Information": {
    "Name": "",
    "Age": "",
    "Gender": "",
    "Date": "",
    "Height": "",
    "Weight": "",
    "BMI": "",
    "SPO2": "",
    "HR": "",
    "B.P": "",
    "TEM": ""
  },
  "Doctor Information": {
    "Name": "",
    "Graduation": "",
    "Hospital Name": "",
    "Location": "",
    "License Number": ""
  },
  "Medication Information": [
    {
      "Medicine Name": "",
      "Dosage": "",
      "Form": "",
      "Quantity": ""
    }
  ],
  "Symptoms": [],
  "Doctor Note": ""
}

Raw OCR Text:
{{raw_ocr_text}}

**Instructions**:
- Extract and populate each field based on the raw text.
- Maintain the original wording and data accuracy; avoid interpretations.
- If a detail is missing or unclear, leave the field empty or use "Not Available."
"""


per_million_usd = {
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
    "gemini-1.5-flash-001": {"input": 0.075, "output": 0.30},
    "gemini-1.5-flash-002": {"input": 0.075, "output": 0.30},
    "gpt-4o-mini": {"input": 0.075, "output": 0.30},
    "gpt-4o": {"input": 0.075, "output": 0.30},
}


def calculate_cost(
    prompt_token_count,
    candidates_token_count,
    model_name="gemini-1.5-flash",
):
    pricing = per_million_usd.get(model_name, None)
    if pricing is None:
        print(f"Unsupported model: {model_name}")
        return None  #

    # Pricing per 1 million tokens
    input_price_per_million = pricing["input"]  # USD
    output_price_per_million = pricing["output"]  # USD

    # Convert token counts to millions
    prompt_tokens_million = prompt_token_count / 1_000_000
    candidates_token_count_million = candidates_token_count / 1_000_000
    input_cost = prompt_tokens_million * input_price_per_million
    output_cost = candidates_token_count_million * output_price_per_million
    total_cost = input_cost + output_cost
    return total_cost


st.title("Medical Prescription Text Extraction")

uploaded_file = st.file_uploader(
    "Choose an image of a medical prescription", type=["jpeg", "jpg", "png"]
)


async def process_image_ocr(image, prompt: str) -> dict:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    model = genai.GenerativeModel(model_name="gemini-1.5-flash-002")
    ocr_start_time = time.time()
    result = model.generate_content(
        contents=[image, "\n\n", prompt]
    )
    ocr_end_time = time.time()
    ocr_time_taken = ocr_end_time - ocr_start_time
    return {
        "response": result.text,
        "time_taken": ocr_time_taken,
        "metadata": result.usage_metadata
    }


async def process_structured_data(ocr_text: str) -> dict:
    model = genai.GenerativeModel(model_name="gemini-1.5-flash-001")
    start_time = time.time()
    result = model.generate_content(
        contents=[prompt_template_extract_structured_data.replace(
            "{{raw_ocr_text}}", ocr_text)]
    )
    end_time = time.time()
    return {
        "response": result.text,
        "time_taken": end_time - start_time,
        "metadata": result.usage_metadata
    }

if uploaded_file is not None:
    uploaded_image_PIL = Image.open(uploaded_file)

    st.sidebar.image(uploaded_image_PIL, caption="Uploaded Prescription")

    with st.spinner("Extracting..."):
        async def run_processing():
            ocr_result = await process_image_ocr(uploaded_image_PIL, prompt_template_extract_ocr)
            structured_result = await process_structured_data(ocr_result["response"])

            total_time = ocr_result["time_taken"] + \
                structured_result["time_taken"]

            ocr_cost = calculate_cost(
                ocr_result["metadata"].prompt_token_count, ocr_result["metadata"].candidates_token_count)
            structured_cost = calculate_cost(
                structured_result["metadata"].prompt_token_count, structured_result["metadata"].candidates_token_count)

            st.json({
                "time": {
                    "ocr_time_taken": ocr_result["time_taken"],
                    "format_time_taken": structured_result["time_taken"],
                    "total_time_taken": total_time
                },
                "cost": {
                    "ocr_cost": ocr_cost,
                    "structured_cost": structured_cost,
                    "total_cost": ocr_cost + structured_cost
                }
            })
            parser = JsonOutputParser()
            result = parser.parse(structured_result["response"])
            st.json(result)

        # Create and run event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(run_processing())
            st.success("Done!")
        finally:
            loop.close()
