"""
Report Generation Service for creating attack analysis reports.
Place this file at: fastapi_backend/services/report_generation_service.py
"""
import json
import requests
from typing import Dict, Any, Optional


class ReportGenerationService:
    """Service for generating attack analysis reports using a local LLM (Ollama)."""
    
    def __init__(
        self,
        ollama_api_url: str = "http://localhost:11434/api/generate",
        default_model: str = "llama3.2",
        timeout: int = 180
    ):
        """
        Initialize the ReportGenerationService.
        
        Args:
            ollama_api_url: URL of the Ollama API endpoint
            default_model: Default LLM model to use
            timeout: Request timeout in seconds
        """
        self.ollama_api_url = ollama_api_url
        self.default_model = default_model
        self.timeout = timeout
        print(f"ReportGenerationService initialized with model: {default_model}")
    
    def _generate_report_prompt(self, graph_summary: Dict[str, Any]) -> str:
        """
        Generate the prompt for the LLM based on the graph summary.
        
        Args:
            graph_summary: Summarized attack graph data
            
        Returns:
            Formatted prompt string
        """
        schema = """
        Your output MUST be ONLY valid JSON and follow this schema exactly:
        {
            "attack_name": "<short formal name of the detected attack type>",
            "attack_summary": "<technical summary of 4-5 sentences describing the attack sequence, methods used, and potential impact>",
            "severity": "<LOW|MEDIUM|HIGH|CRITICAL>",
            "confidence": "<percentage indicating confidence in the analysis>",
            "nist_csf_mapping": {
                "Identify": "<identification stage details - what was discovered>",
                "Protect": "<defensive controls bypassed or missing>",
                "Detect": "<what detections occurred and how>",
                "Respond": "<recommended immediate response actions>",
                "Recover": "<recommended recovery and hardening steps>"
            },
            "attack_timeline": [
                {
                    "step": 1,
                    "action": "<what happened>",
                    "timestamp": "<timestamp if available>"
                }
            ],
            "recommended_actions": [
                "<specific mitigation step 1>",
                "<specific mitigation step 2>",
                "<specific mitigation step 3>"
            ],
            "indicators_of_compromise": [
                "<IOC 1>",
                "<IOC 2>"
            ]
        }
        """
        
        graph_json = json.dumps(graph_summary, indent=2, default=str)
        
        return f"""
You are a cybersecurity incident response analyst with expertise in threat detection and NIST frameworks.

Analyze the following ATTACK GRAPH, which represents relationships between security log events detected by an intrusion detection system. Each node is a security event, and edges represent temporal or causal relationships with attack probabilities.

Your task:
1. Identify the type of attack based on the sequence of events
2. Assess the severity and provide your confidence level
3. Map the attack to the NIST Cybersecurity Framework stages
4. Provide a timeline of the attack sequence
5. Recommend specific, actionable mitigation steps
6. Extract indicators of compromise

Base your reasoning ONLY on the data provided. Output must be valid JSON following the exact schema below.

--- BEGIN ATTACK GRAPH DATA ---
{graph_json}
--- END ATTACK GRAPH DATA ---

{schema}

Remember: Output ONLY the JSON object, no additional text before or after.
"""
    
    def generate_report(
        self,
        graph_summary: Dict[str, Any],
        model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a structured attack report using Ollama (local LLM).
        
        Args:
            graph_summary: Summarized attack graph data
            model_name: LLM model to use (defaults to self.default_model)
            
        Returns:
            Dictionary containing the generated report or error information
        """
        model = model_name or self.default_model
        prompt = self._generate_report_prompt(graph_summary)
        
        print(f"Generating report using model: {model}")
        
        try:
            response = requests.post(
                self.ollama_api_url,
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": True
                },
                timeout=self.timeout,
                stream=True
            )
            
            response.raise_for_status()
            
            # Collect the full response text
            text = ""
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        text += data.get("response", "")
                    except json.JSONDecodeError:
                        continue
            
            text = text.strip()
            
            # Try to parse the JSON response
            try:
                report = json.loads(text)
                print("Report generated successfully")
                return {
                    "success": True,
                    "report": report
                }
            except json.JSONDecodeError:
                # Try to extract JSON from the text
                start = text.find("{")
                end = text.rfind("}") + 1
                
                if start >= 0 and end > start:
                    try:
                        report = json.loads(text[start:end])
                        print("Report extracted from response text")
                        return {
                            "success": True,
                            "report": report
                        }
                    except json.JSONDecodeError:
                        pass
                
                # If JSON extraction fails, return raw output
                print("Warning: Could not parse JSON from LLM response")
                return {
                    "success": False,
                    "error": "Failed to parse JSON from LLM response",
                    "raw_output": text
                }
                
        except requests.exceptions.Timeout:
            print(f"Error: Request timeout after {self.timeout} seconds")
            return {
                "success": False,
                "error": f"Request timeout after {self.timeout} seconds"
            }
        except requests.exceptions.RequestException as e:
            print(f"Error: Request failed - {str(e)}")
            return {
                "success": False,
                "error": f"Request failed: {str(e)}"
            }
        except Exception as e:
            print(f"Error: Unexpected error - {str(e)}")
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}"
            }
    
    def check_ollama_health(self) -> bool:
        """
        Check if Ollama service is available.
        
        Returns:
            True if Ollama is running, False otherwise
        """
        try:
            # Try to ping Ollama API
            base_url = self.ollama_api_url.rsplit('/', 2)[0]  # Get base URL
            response = requests.get(f"{base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception:
            return False