import json
import os
from typing import Dict, Any, Optional, Literal
from dotenv import load_dotenv
import requests
from fastapi_backend.models.schemas import AttackReport
from ollama import chat
from openai import OpenAI


env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=env_path)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OLLAMA_BASE_URL = "http://localhost:11434"

# --- Define LLM Engine Type ---
LLMEngine = Literal["ollama", "groq"]

# --- Constants ---
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

class ReportGenerationService:
    """
    Service for generating attack analysis reports using Ollama or Groq.
    
    Groq relies on the GROQ_API_KEY environment variable.
    """
    
    def __init__(
        self,
        default_ollama_model: str = "llama3.2",
        default_groq_model: str = "openai/gpt-oss-20b",
        timeout: int = 180,
    ):
        """
        Initialize the ReportGenerationService, setting up both LLM clients.
        
        Args:
            default_ollama_model: Default Ollama model to use
            default_groq_model: Default Groq model to use
            timeout: Request timeout in seconds
            ollama_host: URL for the local Ollama service
        """
        self.default_ollama_model = default_ollama_model
        self.default_groq_model = default_groq_model
        self.timeout = timeout
        
        # Initialize Groq client (uses GROQ_API_KEY env var)
        self.groq_client = OpenAI(
            api_key=GROQ_API_KEY,
            base_url=GROQ_BASE_URL,
            timeout=self.timeout
        )
        
        print(f"ReportGenerationService initialized.")
        print(f"- Ollama (Local): {default_ollama_model}")
        print(f"- Groq (API): {default_groq_model}")

    
    def generate_report_prompt(self, graph_summary):
        schema = """
        Your output MUST be ONLY valid JSON and follow this schema exactly:
        {
        "attack_name": "<short formal name of the detected attack type>",
        "attack_summary": "<technical summary of 4-5 sentences>",
        "severity": "<Low/Medium/High/Critical>",
        "confidence": "<score in percentage 0-100>", 
        "nist_csf_mapping": {
            "Identify": "<identification stage details>",
            "Protect": "<defensive controls bypassed>",
            "Detect": "<what detections occurred>",
            "Respond": "<recommended immediate response>",
            "Recover": "<recommended recovery/hardening>"
        },
        "attack_timeline": [
            {"step": 1, "action": "<description of action>", "timestamp": "<if available>"},
            {"step": 2, "action": "<description of action>", "timestamp": "<if available>"},
            "... up to N steps ..."
        ],
        "recommended_actions": [
            "<specific mitigation step 1>",
            "<specific mitigation step 2>",
            "<specific mitigation step 3>"
        ],
        "indicators_of_compromise": [
            "<IOC 1>",
            "<IOC 2>",
            "... up to N IOCs ..."
        ]
        }
        """
        graph_json = json.dumps(graph_summary, indent=2, default=str)
        return f"""
        You are a cybersecurity incident response analyst.
        Analyze the following summarized LOGS GRAPH, which represents relationships
        between security log events detected by an intrusion detection system.
        Provide a formal, concise technical report of the likely attack type and
        mapping to the NIST Cybersecurity Framework.
        Base your reasoning ONLY on the data provided and output strictly formatted JSON.
        
        --- BEGIN GRAPH SUMMARY ---
        {graph_json}
        --- END GRAPH SUMMARY ---

        --- BEGIN EXAMPLE SCHEMA ---
        {schema}
        --- END EXAMPLE SCHEMA ---
        """

    def generate_with_ollama(self, prompt, model_name):
        """Generate a structured report via Ollama (local LLM)."""
        try:
            # Ollama API call
            response = chat(
                model=model_name,
                messages=[{
                    'role': 'user',
                    'content': prompt,
                }],
                # Use Pydantic schema for strict JSON output
                format=AttackReport.model_json_schema(),
            )

            raw_json_output = response['message']['content']
            report_data = json.loads(raw_json_output)
            return report_data

        except requests.exceptions.Timeout:
                # Ollama uses the standard requests library under the hood
                error_msg = f"Ollama Request timeout after {self.timeout} seconds"
                print(f"Error: {error_msg}")
                return {"success": False, "error": error_msg}
        # Catch all other exceptions specific to the Ollama/local connection
        except Exception as e:
            error_msg = f"Ollama/Local request failed: {str(e)}"
            print(f"Error: {error_msg}")
            return {"success": False, "error": error_msg}
    

    def generate_with_groq(self, prompt, model_name):
        try:
            response = self.groq_client.responses.parse(
                model=model_name,
                input=[
                    {"role": "system", "content": prompt},
                ],
                text_format=AttackReport,
            )

            result = response.output_parsed
            report_dict = result.model_dump()
            return report_dict
        
        except json.JSONDecodeError:
            error_msg = f"LLM output was not valid JSON..."
            print(f"Error: {error_msg}")
            return {"success": False, "error": "LLM output was not valid JSON."}
        except Exception as e:
            error_msg = f"Unexpected error during Groq generation: {str(e)}"
            print(f"Error: {error_msg}")
            return {"success": False, "error": error_msg}


    def generate_report(
        self,
        graph_summary: Dict[str, Any],
        llm_engine: LLMEngine = "ollama",
        model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a structured attack report using either Ollama or Groq.
        
        Args:
            graph_summary: Summarized attack graph data
            llm_engine: The LLM provider to use ("ollama" or "groq")
            model_name: LLM model to use (defaults to self.default_ollama_model/default_groq_model)
            
        Returns:
            Dictionary containing the generated report (parsed JSON) or error information
        """

        prompt = self.generate_report_prompt(graph_summary)
        
        if llm_engine == "groq":
            model = model_name or self.default_groq_model
            print(f"Generating report using Groq model: {model}")
            report_data = self.generate_with_groq(prompt, model)                 
            
        elif llm_engine == "ollama":
            model = model_name or self.default_ollama_model
            print(f"Generating report using Ollama model: {model}")
            report_data = self.generate_with_ollama(prompt, model)      
        
        else:
            return {"success": False, "error": f"Unknown LLM engine specified: {llm_engine}"}
    
        return {"success": True, "report": report_data}
    

    async def check_ollama_health(self) -> bool:
        """
        Check if the local Ollama service is available by listing models.
        """
        try:
            response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception as e:
            print(f"Ollama health check failed: {e}")
            return False

    async def check_groq_health(self) -> bool:
        """
        Check if the Groq service is available by listing models.
        """
        try:
            # Attempt a simple, cheap API call (listing models is a good health check)
            self.groq_client.models.list()
            return True
        except Exception:
            # Catches connection errors, timeout, and API key issues
            return False