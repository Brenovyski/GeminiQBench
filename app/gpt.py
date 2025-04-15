import os
import base64
from io import BytesIO
from PIL import Image
from openai import OpenAI

# Initialize the OpenAI client.
client = OpenAI()

def encode_image_from_pil(image: Image.Image) -> str:
    """
    Convert a PIL image to a base64-encoded JPEG string.
    """
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def _prepare_new_payload(
    model_name: str,
    system_instructions: str,
    conversation_history: list,
    user_message: str,
    pil_image: Image.Image = None
):
    """
    Build a SoM-style payload for GPT that obeys the new response structure.

    The payload has the format:
      {
          "model": <model_name>,
          "input": [
              {
                  "role": "user",
                  "content": [
                      { "type": "input_text", "text": "<combined text>" },
                      { "type": "input_image", "image_url": { "url": "data:image/jpeg;base64,<base64_data>" } }   // optional
                  ]
              }
          ],
      }

    The combined text is built by:
      - Prepending the system instructions.
      - Appending all previous conversation turns (each labeled as "User:" or "Assistant:").
      - Finally appending the new user message.
    """
    # Combine conversation history into one text block.
    conversation_text = ""
    if conversation_history:
        for turn in conversation_history:
            # Capitalize the role name for clarity.
            conversation_text += f"{turn['role'].capitalize()}: {turn['content']}\n"
    # Prepend system instructions and append the new user message.
    combined_text = f"{system_instructions}\n{conversation_text}User: {user_message}\n"
    print(combined_text)

    # Build the content array with the text.
    content_array = [
        {"type": "input_text", "text": combined_text.strip()}
    ]
    print(content_array)

    # If there is an image, embed it as an input_image object.
    if pil_image is not None:
        base64_image = encode_image_from_pil(pil_image)
        content_array.append({
            "type": "input_image",
            "image_url": f"data:image/jpeg;base64,{base64_image}"
        })

    # Build and return the final payload.
    if model_name == "gpt-4o" :
        model_name = "chatgpt-4o-latest"  # use the chatgpt version for gpt-4o, this version is more adequated
    payload = {
        "model": model_name,
        "input": [
            {"role": "user", "content": content_array}
        ],
    }
    return payload

def request_gpt_chat(
    model_name: str,
    user_message: str,
    pil_image: Image.Image = None,
    conversation_history: list = None,
    system_instructions: str = (
        "You are a helpful assistant. Your goal here is to extract unknown information from the user about the image given and provide helpful responses. "
        "You can ask questions, provide information, or engage in a conversation. Try to obtain the unknown information from the user and solve what the user wants."
        "IMPORTANT: After you determined the final answer write in front of it: 'Final answer: ' and then refuses to answer any more questions."
    )
) -> str:
    """
    Build a multi-turn conversation in the new SoM-style format:
      - Combine all prior conversation turns and the new user message (prefixed with system instructions)
        into a single text block.
      - Optionally embed a base64 image.
      - Send the request using the new OpenAI SDK call, mimicking:
      
          response = client.responses.create(
              model="gpt-4o",
              input=[ { "role": "user", "content": [ ... ] } ],
          )
    
    Returns the assistant's output text.
    """
    if conversation_history is None:
        conversation_history = []

    # Prepare the payload according to the new format.
    payload = _prepare_new_payload(
        model_name=model_name,
        system_instructions=system_instructions,
        conversation_history=conversation_history,
        user_message=user_message,
        pil_image=pil_image
    )
    print(payload['model']),
    # Use the new response structure.
    response = client.responses.create(  # For debugging, print the payload.
        model=payload["model"],
        input=payload["input"],
    )

    return response.output_text
