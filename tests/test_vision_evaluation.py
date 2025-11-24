import pytest
import os
import numpy as np
import cv2
from da3d.evaluation.vision_agent import VisionAgent

# Define a fixture to create a dummy image for testing
@pytest.fixture
def dummy_image(tmp_path):
    image_path = tmp_path / "test_image.png"
    # Create a simple 100x100 red image
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:] = (0, 0, 255)
    cv2.imwrite(str(image_path), img)
    return str(image_path)

@pytest.mark.vision
def test_vision_agent_mock_mode(dummy_image):
    """Test that the agent works in mock mode without an API key."""
    # Force mock mode
    agent = VisionAgent(mock=True)
    response = agent.evaluate_image(dummy_image, "Is this a red image?")
    
    assert "[MOCK RESPONSE]" in response
    assert "valid" in response

@pytest.mark.vision
def test_vision_agent_initialization_with_env():
    """Test initialization logic (mock vs real) based on env vars."""
    # Save original env
    original_key = os.environ.get("OPENAI_API_KEY")
    
    try:
        # Case 1: No API key -> Should default to Mock
        if "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]
        
        agent = VisionAgent()
        assert agent.mock is True
        
        # Case 2: With API key -> Should be Real (unless mock=True passed)
        os.environ["OPENAI_API_KEY"] = "sk-dummy-key"
        # We can't easily test successful init of OpenAI client without a real key or mocking the library,
        # but we can check if it *tries* to be real.
        # However, VisionAgent init will fail if we pass a dummy key and it tries to validate? 
        # Actually OpenAI client doesn't validate on init, only on call.
        # But we need 'openai' installed.
        
        try:
            import openai
            agent_real = VisionAgent()
            assert agent_real.mock is False
            assert agent_real.api_key == "sk-dummy-key"
        except ImportError:
            pass # Skip if openai not installed

    finally:
        # Restore env
        if original_key:
            os.environ["OPENAI_API_KEY"] = original_key
        elif "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]

@pytest.mark.vision
def test_vision_agent_openrouter_config():
    """Test that base_url and model are correctly set."""
    agent = VisionAgent(
        api_key="sk-dummy", 
        base_url="https://openrouter.ai/api/v1", 
        model="google/gemini-2.0-flash-exp:free",
        mock=False
    )
    
    assert agent.base_url == "https://openrouter.ai/api/v1"
    assert agent.model == "google/gemini-2.0-flash-exp:free"
