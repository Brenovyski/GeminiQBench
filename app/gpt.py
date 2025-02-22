import os
import requests
import base64
from io import BytesIO
from PIL import Image

# Load your OpenAI API Key from environment variables
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY is not set in environment variables.")

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

def _prepare_som_payload(
    model_name: str,
    system_instructions: str,
    conversation_history: list,
    user_message: str,
    pil_image: Image.Image = None
):
    """
    Build a SoM-style payload for GPT.
    We'll treat the entire conversation + new user message as a single
    'user' content array with {type: 'text'} + optional {type: 'image_url'}.

    The conversation is concatenated into one text block, then appended
    with the new user message. The image is embedded as base64 in the 'image_url'.
    """

    # 1) Combine the entire conversation into a single text block
    #    For each turn, label it as 'User:' or 'Assistant:'.
    conversation_text = ""
    if conversation_history:
        for turn in conversation_history:
            if turn["role"] == "assistant":
                conversation_text += f"Assistant: {turn['content']}\n"
            else:
                conversation_text += f"User: {turn['content']}\n"

    # 2) Append the new user message
    conversation_text += f"User: {user_message}\n"

    # 3) Build the user content array
    #    We have "type": "text" for the conversation text
    #    If we have an image, we add "type": "image_url" with base64 data
    user_content = [
        {
            "type": "text",
            "text": conversation_text.strip()
        }
    ]

    # 4) Optionally embed the image in base64
    if pil_image is not None:
        buffered = BytesIO()
        pil_image.save(buffered, format="JPEG")
        base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
        user_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
        })

    # 5) Construct the final SoM-style payload
    #    - The system message is an array containing your system_instructions
    #    - The user message is an array with {type: text} and optionally {type: image_url}
    payload = {
        "model": model_name,  # e.g., "chatgpt-4o-latest"
        "messages": [
            {
                "role": "system",
                "content": [system_instructions]
            },
            {
                "role": "user",
                "content": user_content
            }
        ],
        # If your model supports max_tokens, you can add it here:
        "max_tokens": 800
    }
    return payload

def request_gpt_chat(
    model_name: str,
    user_message: str,
    pil_image: Image.Image = None,
    conversation_history: list = None,
    system_instructions: str = (
        "You are a helpful assistant. Your goal here is to extract unknown information "
        "from the user about the image given and provide helpful responses to the user. "
        "You can ask questions, provide information, or engage in a conversation. "
        "Try to obtain the unknown information from the user and solve what the user wants."
    )
) -> str:
    """
    Build a multi-turn conversation in SoM style by:
      - Combining all conversation history into one text block
      - Appending the new user message
      - Optionally embedding a base64 image
      - Sending the request to a SoM-compatible endpoint (like "chatgpt-4o-latest").

    Note: This lumps the entire conversation into a single user message array, which
    differs from standard ChatCompletion multi-turn usage. But it matches the SoM structure.
    """

    if conversation_history is None:
        conversation_history = []

    # 1) Create the SoM-style payload
    payload = _prepare_som_payload(
        model_name=model_name,
        system_instructions=system_instructions,
        conversation_history=conversation_history,
        user_message=user_message,
        pil_image=pil_image
    )

    # 2) Send the request
    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers,
        json=payload
    )
    data = response.json()

    # 3) Check for errors
    if "error" in data:
        raise ValueError(f"OpenAI API Error: {data['error']}")

    # 4) Return the assistant's text
    return data["choices"][0]["message"]["content"]
