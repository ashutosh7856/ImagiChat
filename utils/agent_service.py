# from pathlib import Path

# from agno.agent import Agent
# from agno.media import Image
# from agno.models.openai import OpenAIChat
# import os
# from dotenv import load_dotenv

# load_dotenv()

# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# agent = Agent(
#     model=OpenAIChat(
#         id="gpt-4o-mini",
#         api_key=OPENAI_API_KEY
#     ),
#     markdown=True,
# )

# image_path = Path(__file__).parent.parent.joinpath("images", "image1.JPG")
# agent.print_response(
#     "Read the question carefully and select one option out of 4 that is there in the image and give me total token used.",
#     images=[Image(filepath=image_path)],
# )



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


def process_image_question(image_name: str, question: str):
    """
    Takes an image filename and question, processes it via the agent,
    and returns the cleaned response text + token usage.
    """
    image_path = Path(__file__).parent.parent.joinpath("images", image_name)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Run the model
    run_output = agent.run(
        question,
        images=[Image(filepath=image_path)]
    )

    # Extract readable text
    response_text = getattr(run_output, "output_text", None)
    if not response_text:
        response_text = str(run_output.content) if hasattr(run_output, "content") else str(run_output)

    # Extract tokens if available
    tokens_used = None
    if hasattr(run_output, "metrics") and getattr(run_output.metrics, "total_tokens", None):
        tokens_used = run_output.metrics.total_tokens
    elif hasattr(run_output, "usage") and isinstance(run_output.usage, dict):
        tokens_used = run_output.usage.get("total_tokens")

    return {
        "response": response_text,
        "tokens_used": tokens_used
    }
