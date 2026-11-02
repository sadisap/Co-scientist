
import os

from dotenv import load_dotenv
from ollama import chat
from openai import OpenAI
from pydantic import BaseModel
from core.usage import record_usage

load_dotenv()

MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "ollama")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:4b")
OPENAI_MODEL = os.getenv("OPENAI_MODEL")


def generate_structured(
    system_prompt: str,
    user_prompt: str,
    response_model: type[BaseModel],
):
    if MODEL_PROVIDER == "ollama":
        response = chat(
            model=OLLAMA_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            format=response_model.model_json_schema(),
            options={"temperature": 0.7},
        )

        return response_model.model_validate_json(
            response.message.content
        )

    elif MODEL_PROVIDER == "openai":
        if not OPENAI_MODEL:
            raise ValueError("OPENAI_MODEL is missing from .env")

        client = OpenAI()

        response = client.responses.parse(
            model=OPENAI_MODEL,
            input=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            text_format=response_model,
        )
        record_usage(response)

        if response.output_parsed is None:
            raise RuntimeError(
                "OpenAI returned no structured output"
            )

        return response.output_parsed

    else:
        raise ValueError(
            f"Unknown MODEL_PROVIDER: {MODEL_PROVIDER}"
        )
