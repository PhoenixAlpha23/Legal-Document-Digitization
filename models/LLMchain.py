import streamlit as st
import requests
import json
from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema.runnable import RunnablePassthrough
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
import re

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

prompt_template = """
You are an expert in error correction and entity extraction, with expertise in multilingual processing (English, हिन्दी, मराठी).
Analyze the given text and perform the following tasks:
1. Named Entity Recognition: Identify key roles such as:
   - PERSON: Names of individuals (e.g., Mahesh, Suresh, etc.)
   - ORG: Organizations (Issuing Authority, Companies involved)
   - DATE: Important dates (Issue Date, Expiry Date, Agreement Date)
   - LOC: Locations mentioned in the document
   - OTHER: Any other relevant entities (e.g., Contract Number, Registration ID)
2. Summarization: Provide a brief summary of the document covering:
   - Document Type (Certificate, Agreement, Contract, etc.)
   - Purpose of the document
   - Key points (Validity, Terms, Clauses)

**Text:**
{text}

**IMPORTANT RULES**
1. The targeted domain of the text is legal documentation
2. CRITICAL: ALL output fields (document_type, summary, etc.) MUST be in the SAME LANGUAGE AND SCRIPT as the input text
3. If input is in Hindi script (देवनागरी), respond entirely in Hindi script
4. If input is in Marathi, respond entirely in Marathi
5. If input is in English, respond in English

Respond in this exact JSON format:
{{
    "entities Recognised": [
        {{
            "text": "extracted entity",
            "type": "entity type (PERSON, ORG, DATE, LOC, OTHER)"
        }}
    ],
    "document_type": "Detected document type (same script as input)",
    "summary": "Brief summary of the document (same script as input)"
}}

IMPORTANT: Ensure your response is only the JSON object above, with no additional text, preamble, or explanation.
"""

class GroqLLM(LLM, BaseModel):
    api_key: str = GROQ_API_KEY
    model_name: str = "gemma2-9b-it"
    temperature: float = 0.0
    max_tokens: int = 1024

    @property
    def _llm_type(self) -> str:
        return "groq"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        if not self.api_key:
            return '{"error": "API key not found in environment variables"}'
            
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }

        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                json=payload,
                headers=headers
            )

            if response.status_code != 200:
                return f'{{"error": "Groq API error: {response.status_code} - {response.text}"}}'

            response_json = response.json()
            return response_json["choices"][0]["message"]["content"]
            
        except requests.exceptions.RequestException as e:
            return f'{{"error": "API request failed: {str(e)}"}}'
        except Exception as e:
            return f'{{"error": "Unknown error: {str(e)}"}}'

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }

def extract_json_from_text(text):
    """Extract JSON from text in case the LLM adds extra content"""
    # Try to find JSON pattern within the text
    json_match = re.search(r'(\{.*\})', text, re.DOTALL)
    if json_match:
        return json_match.group(1)
    return text

@st.cache_data(ttl=3600)  # Cache for 1 hour
def process_legal_text(text: str) -> Dict:
    try:
        MAX_CHARS = 10000 # max limit for text
        if len(text) > MAX_CHARS:
            text = text[:MAX_CHARS] + "..."

        if not GROQ_API_KEY:
            return {"error": "API key not found in environment variables"}

        llm = GroqLLM()
        prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
        chain = prompt | llm
        
        # Get raw response from LLM
        raw_response = chain.invoke({"text": text})
        
        # Log the raw response for debugging
        print(f"Raw LLM response: {raw_response}")
        
        # Clean up response to extract JSON
        json_str = extract_json_from_text(raw_response)
        
        # Try to parse as JSON
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            # If can't parse the JSON, show  friendly error and the raw response
            return {
                "error": f"Could not parse LLM response as JSON: {str(e)}",
                "raw_response": raw_response[:500] + ("..." if len(raw_response) > 500 else "")
            }

    except Exception as e:
        return {"error": f"Processing failed: {str(e)}"}
