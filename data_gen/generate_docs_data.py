import datetime
import random
import json
import os
import time
import threading
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load environment variables
load_dotenv(".env")

# OpenAI API Key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Number of documents to generate
NUM_DOCUMENTS = 5

# Define document categories
DOCUMENT_TEMPLATES = [
    {
        "name": "Survey_Process",
        "prompt": """
            Write a detailed **internal process document** for how patient surveys are conducted in a healthcare business.
            Include details on:
            - The overall workflow for issuing and collecting survey responses
            - The technology stack (Twilio, Google Forms, internal CRM, etc.)
            - How patient data is handled for privacy and compliance (HIPAA, GDPR)
            - How survey effectiveness is measured and improved over time
            - Any known challenges or pain points in the process

            The document should be **at least 3 pages long** and structured in a formal business document style.
        """,
    },
    {
        "name": "Survey_Agreements",
        "prompt": """
            Generate a **contract agreement** between a healthcare provider and a survey software vendor.
            This should include:
            - The scope of services provided
            - Data ownership and privacy clauses
            - Cost structure and payment terms
            - Termination conditions
            - Responsibilities of both parties

            The document should **simulate a real contract**, including standard legalese.
        """,
    },
    {
        "name": "Survey_Rates",
        "prompt": """
            Generate an internal **rate card document** that lists the negotiated rates for survey-related services.
            The document should include:
            - The per-unit cost for each communication method (SMS, Email, Phone)
            - Bulk discount thresholds
            - Additional fees for special services (e.g., survey analysis, urgent surveys)
            - Effective date of the rates
            - Any pending rate adjustments in the pipeline

            Format the document as an **official pricing memo** with proper tables and sections.
        """,
    },
    {
        "name": "Compliance_Guidelines",
        "prompt": """
            Write an **internal compliance document** outlining the legal and ethical considerations of handling patient feedback.
            The document should cover:
            - **HIPAA compliance** for survey data storage
            - Guidelines for **anonymous vs. identifiable** patient surveys
            - Restrictions on contacting patients (opt-in/opt-out policies)
            - How patient complaints should be escalated and resolved
            - A FAQ section addressing common compliance concerns

            The document should be **thorough and at least 3 pages long**.
        """,
    },
    {
        "name": "Survey_Optimization_Strategies",
        "prompt": """
            Write a **strategic business report** analyzing ways to optimize patient survey responses.
            Include:
            - Analysis of response trends across different communication methods
            - Recommendations on which methods work best for different demographics
            - Cost-benefit analysis of various survey distribution methods
            - Use of AI in optimizing response rates
            - Case studies or examples from real-world implementations

            This should be **a long-form strategy report** designed for an executive audience.
        """,
    },
]

# System message for OpenAI
SYSTEM_MESSAGE = """
You generate long-form business documents. Each response should be **a single, standalone document** formatted for professional use.
The document should be **detailed, structured, and at least 3 pages long**.
Do not return JSON or structured data, only raw text.
"""


def generate_document(prompt_obj):
    """
    Generate a long-form document using OpenAI's GPT model.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": prompt_obj["prompt"]},
            ],
            temperature=0.7,
            max_tokens=4000,  # Generate long text (adjust as needed)
        )
        document_text = response.choices[0].message.content.strip()
        return {"Document_Name": prompt_obj["name"], "Text": document_text}
    except Exception as e:
        print(f"Error generating document '{prompt_obj['name']}': {e}")
        return None


def rate_limited_generate_document(prompt_obj):
    """
    Rate-limited document generation to avoid API throttling.
    """
    with rate_limited_generate_document.lock:
        current_time = time.time()
        if current_time - rate_limited_generate_document.last_request_time < 1:
            time.sleep(
                1 - (current_time - rate_limited_generate_document.last_request_time)
            )
        rate_limited_generate_document.last_request_time = time.time()

    return generate_document(prompt_obj)


rate_limited_generate_document.last_request_time = 0
rate_limited_generate_document.lock = threading.Lock()


# Generate documents concurrently
def main():
    generated_documents = []

    with ThreadPoolExecutor(max_workers=5) as executor:  # Adjust as needed
        futures = [
            executor.submit(rate_limited_generate_document, doc)
            for doc in DOCUMENT_TEMPLATES
        ]

        for future in as_completed(futures):
            result = future.result()
            if result:
                generated_documents.append(result)
                print(f"Generated document: {result['Document_Name']}")

    # Convert to DataFrame
    df = pd.DataFrame(generated_documents)

    # Save to CSV
    df.to_csv("data/synthetic_documents.csv", index=False)

    print("✅ Documents saved as synthetic_documents.csv")


if __name__ == "__main__":
    main()
