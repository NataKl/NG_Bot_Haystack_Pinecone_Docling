from typing import Dict

from openai import OpenAI

from assistant_tools import get_random_dog_fact, get_random_dog_image_with_description


class ToolsService:
    def __init__(self, openai_client: OpenAI, vision_model: str) -> None:
        self.openai_client = openai_client
        self.vision_model = vision_model

    def get_dog_fact(self) -> str:
        return get_random_dog_fact()

    def get_dog_photo(self) -> Dict[str, str]:
        return get_random_dog_image_with_description(
            openai_client=self.openai_client,
            vision_model=self.vision_model,
        )
