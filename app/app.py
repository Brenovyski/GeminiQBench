import streamlit as st
from PIL import Image
import gemini  # the gemini.py module

def main():
    st.set_page_config(layout="wide")
    st.title("GeminiQBench: Gemini 2.0 Object Detection + Multi-Turn Chat")

    # ----------------------------------------------------------------
    # Session State Initialization
    # ----------------------------------------------------------------
    if "conversation_history" not in st.session_state:
        st.session_state["conversation_history"] = []
    if "masked_image" not in st.session_state:
        st.session_state["masked_image"] = None
    if "model_choice" not in st.session_state:
        st.session_state["model_choice"] = "gemini-1.5-pro"

    # Track previous values to detect changes
    if "prev_model_choice" not in st.session_state:
        st.session_state["prev_model_choice"] = st.session_state["model_choice"]
    if "prev_image_name" not in st.session_state:
        st.session_state["prev_image_name"] = None  # or an empty string

    # ----------------------------------------------------------------
    # 1) Model Selection
    # ----------------------------------------------------------------
    model_options = ["gemini-1.5-pro", "gemini-2.0-flash", "gpt-4o", "o1"]
    st.session_state["model_choice"] = st.selectbox(
        "Select a model to chat with:",
        model_options,
        index=model_options.index(st.session_state["model_choice"])
    )

    # ----------------------------------------------------------------
    # 2) Image Upload
    # ----------------------------------------------------------------
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    current_image_name = uploaded_file.name if uploaded_file is not None else None

    # ----------------------------------------------------------------
    # Check for Changes & Reset Chat if Needed
    # ----------------------------------------------------------------

    # (A) Model changed
    if st.session_state["model_choice"] != st.session_state["prev_model_choice"]:
        st.session_state["conversation_history"] = []
        st.session_state["masked_image"] = None
        st.session_state["prev_model_choice"] = st.session_state["model_choice"]
        st.session_state["prev_image_name"] = current_image_name

    # (B) Image changed (or removed)
    elif current_image_name != st.session_state["prev_image_name"]:
        st.session_state["conversation_history"] = []
        st.session_state["masked_image"] = None
        st.session_state["prev_image_name"] = current_image_name

    # ----------------------------------------------------------------
    # 3) If we have an image, display it and generate the masked image
    # ----------------------------------------------------------------
    if uploaded_file is not None:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original Image")
            original_image = Image.open(uploaded_file)
            st.image(original_image, use_container_width=True)

        with col2:
            st.subheader("Masked Image (Gemini 2.0 Object Detection)")
            if st.session_state["masked_image"] is None:
                with st.spinner("Processing image with Gemini 2.0..."):
                    masked = gemini.generate_masked_image(
                        original_image,
                        prompt="Detect all objects in the image."
                    )
                st.session_state["masked_image"] = masked
                st.image(masked, caption="Detected Objects", use_container_width=True)
            else:
                st.image(st.session_state["masked_image"], caption="Detected Objects", use_container_width=True)

    st.write("---")
    st.subheader(f"Chat with {st.session_state['model_choice']}")

    # ----------------------------------------------------------------
    # 4) Display the Existing Conversation
    # ----------------------------------------------------------------
    for turn in st.session_state["conversation_history"]:
        if turn["role"] == "user":
            with st.chat_message("user"):
                st.write(turn["content"])
        else:  # assistant
            with st.chat_message("assistant"):
                st.write(turn["content"])

    # ----------------------------------------------------------------
    # 5) Chat Input
    # ----------------------------------------------------------------
    user_input = st.chat_input("Type your message here...")

    if user_input:
        # If no masked image, we can't proceed
        if not uploaded_file or st.session_state["masked_image"] is None:
            st.warning("Please upload an image and generate the masked image first.")
            st.stop()

        # Immediately show user's message
        with st.chat_message("user"):
            st.write(user_input)

        # Assistant placeholder
        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.write("Thinking...")

        # Determine if we attach the masked image only on the FIRST user message
        #user_messages_count = sum(1 for msg in st.session_state["conversation_history"] if msg["role"] == "user")
        
        image_to_send = st.session_state["masked_image"]
        
        # Call the model
        response_text = gemini.request_gemini_chat(
            model_name=st.session_state["model_choice"],
            user_message=user_input,
            pil_image=image_to_send,
            conversation_history=st.session_state["conversation_history"],
            system_instructions=(
                "You are a helpful assistant. "
                "Ask clarifying questions based on the image (if provided) and the user's messages."
            )
        )

        # Update placeholder with final response
        placeholder.write(response_text)

        # Save user & assistant messages
        st.session_state["conversation_history"].append({"role": "user", "content": user_input})
        st.session_state["conversation_history"].append({"role": "assistant", "content": response_text})

        st.stop()

if __name__ == "__main__":
    main()
