from ollama import chat
from pydantic import BaseModel


MODEL = "qwen3:4b"


def generate_structured(
    system_prompt: str,
    user_prompt: str,
    response_model: type[BaseModel],
):
    """
    Send a prompt to a local Ollama model and return
    a validated Pydantic object.
    """

    response = chat(
        model=MODEL,
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
        options={
            "temperature": 0.7,
        },
    )

    return response_model.model_validate_json(
        response.message.content
    )