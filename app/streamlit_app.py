import time
import streamlit as st
from PIL import Image
import google.generativeai as genai
from src.prompts.prompts import prompt_template_extract_ocr, prompt_template_extract_structured_datas
import asyncio
from langchain_core.output_parsers import JsonOutputParser
from src.utils import calculate_cost
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(layout="wide")

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
                "ocr_time_taken": ocr_result["time_taken"],
                "format_time_taken": structured_result["time_taken"],
                "total_time_taken": total_time,
                "ocr_cost": ocr_cost,
                "structured_cost": structured_cost,
                "total_cost": ocr_cost + structured_cost
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
