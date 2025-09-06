"""
Ollama client for CSV Insight app.
Handles communication with local Ollama instance for LLM operations.
"""

import requests
import json
import time
from typing import Generator, Dict, Any, Optional, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OllamaClientError(Exception):
    """Custom exception for Ollama client errors."""
    pass


def _extract_error_message(response: requests.Response) -> str:
    """Extract detailed error message from an HTTP response."""
    try:
        data = response.json()
        return data.get("error") or response.text
    except ValueError:
        return response.text


class OllamaClient:
    """Client for interacting with local Ollama instance."""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.timeout = 30
    
    def is_available(self) -> bool:
        """Check if Ollama is running and accessible."""
        try:
            response = self.session.get(f"{self.base_url}/api/tags")
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Ollama not available: {e}")
            return False
    
    def get_models(self) -> List[Dict[str, Any]]:
        """Get list of available models."""
        try:
            response = self.session.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                data = response.json()
                return data.get("models", [])
            return []
        except Exception as e:
            logger.error(f"Failed to get models: {e}")
            return []
    
    def generate(self, prompt: str, system: str = "", model: str = "llama3.1:8b", 
                temperature: float = 0.2, top_p: float = 0.9, 
                max_tokens: int = 2048) -> Generator[str, None, None]:
        """
        Generate text using Ollama's streaming API.
        
        Args:
            prompt: User prompt
            system: System prompt
            model: Model name
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            max_tokens: Maximum tokens to generate
            
        Yields:
            Text chunks as they're generated
        """
        url = f"{self.base_url}/api/generate"
        
        payload = {
            "model": model,
            "prompt": prompt,
            "system": system,
            "stream": True,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "num_predict": max_tokens
            }
        }
        
        try:
            response = self.session.post(url, json=payload, stream=True)
            try:
                response.raise_for_status()
            except requests.exceptions.HTTPError as http_err:
                message = _extract_error_message(http_err.response)
                logger.error(f"Ollama API error: {message}")
                raise OllamaClientError(message) from http_err

            full_response = ""
            start_time = time.time()

            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line.decode('utf-8'))

                        if 'response' in data:
                            chunk = data['response']
                            full_response += chunk
                            yield chunk

                        if data.get('done', False):
                            break

                    except json.JSONDecodeError:
                        continue

            elapsed_time = time.time() - start_time
            logger.info(f"Generated {len(full_response)} characters in {elapsed_time:.2f}s")

        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            raise OllamaClientError(f"Failed to communicate with Ollama: {e}") from e
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise OllamaClientError(str(e)) from e
    
    def generate_sync(self, prompt: str, system: str = "", model: str = "llama3.1:8b",
                      temperature: float = 0.2, top_p: float = 0.9,
                      max_tokens: int = 2048) -> str:
        """
        Generate text synchronously (non-streaming).
        
        Args:
            prompt: User prompt
            system: System prompt
            model: Model name
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            max_tokens: Maximum tokens to generate
            
        Returns:
            Complete generated text
        """
        url = f"{self.base_url}/api/generate"
        
        payload = {
            "model": model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "num_predict": max_tokens
            }
        }
        
        try:
            response = self.session.post(url, json=payload)
            try:
                response.raise_for_status()
            except requests.exceptions.HTTPError as http_err:
                message = _extract_error_message(http_err.response)
                logger.error(f"Ollama API error: {message}")
                raise OllamaClientError(message) from http_err

            data = response.json()
            return data.get('response', '')

        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            raise OllamaClientError(f"Failed to communicate with Ollama: {e}") from e
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise OllamaClientError(str(e)) from e
    
    def embeddings(self, text: str, model: str = "llama3.1:8b") -> Optional[List[float]]:
        """
        Generate embeddings for text.
        
        Args:
            text: Text to embed
            model: Model name
            
        Returns:
            Embedding vector or None if failed
        """
        url = f"{self.base_url}/api/embeddings"
        
        payload = {
            "model": model,
            "prompt": text
        }
        
        try:
            response = self.session.post(url, json=payload)
            try:
                response.raise_for_status()
            except requests.exceptions.HTTPError as http_err:
                message = _extract_error_message(http_err.response)
                logger.error(f"Embeddings request failed: {message}")
                raise OllamaClientError(message) from http_err

            data = response.json()
            return data.get('embedding', [])

        except requests.exceptions.RequestException as e:
            logger.error(f"Embeddings request failed: {e}")
            raise OllamaClientError(f"Failed to communicate with Ollama: {e}") from e
        except Exception as e:
            logger.error(f"Embeddings generation failed: {e}")
            raise OllamaClientError(str(e)) from e
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check and return status information."""
        status = {
            "available": False,
            "models": [],
            "error": None
        }
        
        try:
            if self.is_available():
                status["available"] = True
                status["models"] = self.get_models()
            else:
                status["error"] = "Ollama not accessible"
        except Exception as e:
            status["error"] = str(e)
        
        return status


# Convenience functions for backward compatibility
def generate(prompt: str, system: str = "", model: str = "llama3.1:8b", 
            temperature: float = 0.2, top_p: float = 0.9) -> Generator[str, None, None]:
    """Convenience function for text generation."""
    client = OllamaClient()
    yield from client.generate(prompt, system, model, temperature, top_p)


def generate_sync(prompt: str, system: str = "", model: str = "llama3.1:8b",
                  temperature: float = 0.2, top_p: float = 0.9) -> str:
    """Convenience function for synchronous text generation."""
    client = OllamaClient()
    return client.generate_sync(prompt, system, model, temperature, top_p)


def embeddings(text: str, model: str = "llama3.1:8b") -> Optional[List[float]]:
    """Convenience function for embeddings."""
    client = OllamaClient()
    return client.embeddings(text, model)
