from pathlib import Path

from agno.agent import Agent
from agno.media import Image
from agno.models.openai import OpenAIChat
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

agent = Agent(
    model=OpenAIChat(
        id="gpt-4o-mini",
        api_key=OPENAI_API_KEY
    ),
    markdown=True,
)

image_path = Path(__file__).parent.parent.joinpath("images", "image1.JPG")
agent.print_response(
    "Read the question carefully and select one option out of 4 that is there in the image and give me total token used.",
    images=[Image(filepath=image_path)],
)