import os
import base64
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any

try:
    from openai import OpenAI, APIError, RateLimitError
except ImportError:
    OpenAI = None
    APIError = None
    RateLimitError = None

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

class VisionAgent:
    def __init__(
        self, 
        api_key: Optional[str] = None, 
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        mock: bool = False
    ):
        if load_dotenv:
            load_dotenv()

        self.mock = mock
        # Auto-enable mock if no API key is present and not explicitly provided
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        
        if not self.api_key and not self.mock:
            print("Warning: No API key found. Defaulting to Mock Mode.")
            self.mock = True

        if not self.mock:
            if OpenAI is None:
                raise ImportError("The 'openai' package is required for VisionAgent. Please install it with 'pip install openai'.")
            
            # Default to OpenRouter if configured via env, otherwise OpenAI default
            default_base_url = os.environ.get("OPENAI_BASE_URL")
            self.base_url = base_url or default_base_url
            
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        
        # Default model logic
        self.model = model or os.environ.get("OPENAI_MODEL") or "gpt-4o"

    def _encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def evaluate_image(self, image_path: str, prompt: str) -> str:
        """
        Evaluates an image using a VLM based on the provided prompt.
        
        Args:
            image_path: Path to the image file.
            prompt: The question or instruction for the VLM.
            
        Returns:
            The text response from the VLM.
        """
        if not Path(image_path).exists():
            return f"Error: Image not found at {image_path}"

        if self.mock:
            return (
                "[MOCK RESPONSE] The image appears to be valid. "
                "Geometry is coherent. No flying pixels detected. "
                "This is a simulated response for testing purposes."
            )

        base64_image = self._encode_image(image_path)

        max_retries = 5
        base_delay = 5

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{base64_image}"
                                    },
                                },
                            ],
                        }
                    ],
                    max_tokens=300,
                )
                return response.choices[0].message.content
            except RateLimitError:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    print(f"Rate limit hit. Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    return "Error: Rate limit exceeded after multiple retries."
            except APIError as e:
                return f"Error calling API: {e}"
            except Exception as e:
                return f"Unexpected error: {e}"
