
from pathlib import Path
from agno.agent import Agent
from agno.media import Image
from agno.models.openai import OpenAIChat
from utils.config import OPENAI_API_KEY


agent = Agent(
    model=OpenAIChat(
        id="gpt-4o-mini",
        api_key=OPENAI_API_KEY
    ),
    markdown=True,
)

def process_image_question(image_path: Path, question: str):
    """
    Takes an uploaded image path + question, processes it via the agent,
    and returns the response text + token usage count.
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Run the AI model with image input
    run_output = agent.run(
        question,
        images=[Image(filepath=image_path)]
    )

    # Extract readable response text
    response_text = getattr(run_output, "output_text", None)
    if not response_text:
        if hasattr(run_output, "content"):
            response_text = str(run_output.content)
        else:
            response_text = str(run_output)

    # Extract token count if available
    tokens_used = None
    if hasattr(run_output, "metrics") and getattr(run_output.metrics, "total_tokens", None):
        tokens_used = run_output.metrics.total_tokens
    elif hasattr(run_output, "usage") and isinstance(run_output.usage, dict):
        tokens_used = run_output.usage.get("total_tokens")

    return {
        "response": response_text,
        "tokens_used": tokens_used
    }
