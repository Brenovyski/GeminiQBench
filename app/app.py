import streamlit as st
from PIL import Image
# Import Gemini for object detection & chat, and GPT for chat
import gemini
import gpt

def main():
    st.set_page_config(layout="wide")
    st.title("GeminiQBench: Multi-Model with Gemini Object Detection + 512px Masked Image")

    # ----------------------------------------------------------------
    # Session State Initialization
    # ----------------------------------------------------------------
    if "conversation_history" not in st.session_state:
        st.session_state["conversation_history"] = []
    if "masked_image" not in st.session_state:
        st.session_state["masked_image"] = None
    if "model_choice" not in st.session_state:
        st.session_state["model_choice"] = "gemini-1.5-pro"

    # Track previous model & image to detect changes
    if "prev_model_choice" not in st.session_state:
        st.session_state["prev_model_choice"] = st.session_state["model_choice"]
    if "prev_image_name" not in st.session_state:
        st.session_state["prev_image_name"] = None

    # ----------------------------------------------------------------
    # 1) Model Selection
    # ----------------------------------------------------------------
    model_options = ["gemini-1.5-pro", "gemini-2.0-flash", "gpt-4o"]
    st.session_state["model_choice"] = st.selectbox(
        "Select a model to chat with:",
        model_options,
        index=model_options.index(st.session_state["model_choice"])
    )

    # ----------------------------------------------------------------
    # 2) Image Upload
    # ----------------------------------------------------------------
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    current_image_name = uploaded_file.name if uploaded_file else None

    # ----------------------------------------------------------------
    # 3) Check for Changes & Reset Chat if Needed
    # ----------------------------------------------------------------
    # (A) If model changed
    if st.session_state["model_choice"] != st.session_state["prev_model_choice"]:
        st.session_state["conversation_history"] = []
        st.session_state["prev_model_choice"] = st.session_state["model_choice"]
        st.session_state["prev_image_name"] = current_image_name

    # (B) If image changed (or removed)
    elif current_image_name != st.session_state["prev_image_name"]:
        st.session_state["conversation_history"] = []
        st.session_state["masked_image"] = None
        st.session_state["prev_image_name"] = current_image_name

    # ----------------------------------------------------------------
    # 4) If we have an image, display it & do Gemini object detection
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
                    # 1) Run object detection
                    masked = gemini.generate_masked_image(
                        original_image,
                        prompt="Detect all objects in the image."
                    )
                    # 2) Standardize to 512px height
                    max_height = 512
                    w, h = masked.size
                    if h > max_height:
                        ratio = max_height / float(h)
                        new_w = int(w * ratio)
                        masked = masked.resize((new_w, max_height), Image.Resampling.LANCZOS)

                st.session_state["masked_image"] = masked
                st.image(masked, caption="Detected Objects (512px high)", use_container_width=True)
            else:
                st.image(st.session_state["masked_image"], caption="Detected Objects (512px high)", use_container_width=True)

    st.write("---")
    st.subheader(f"Chat with {st.session_state['model_choice']}")

    # ----------------------------------------------------------------
    # 5) Display the Existing Conversation
    # ----------------------------------------------------------------
    for turn in st.session_state["conversation_history"]:
        if turn["role"] == "user":
            with st.chat_message("user"):
                st.write(turn["content"])
        else:  # assistant
            with st.chat_message("assistant"):
                st.write(turn["content"])

    # ----------------------------------------------------------------
    # 6) Chat Input
    # ----------------------------------------------------------------
    user_input = st.chat_input("Type your message here...")

    if user_input:
        # If we have no masked image, we can't proceed
        if not uploaded_file or st.session_state["masked_image"] is None:
            st.warning("Please upload an image first.")
            st.stop()

        # Immediately show user's message
        with st.chat_message("user"):
            st.write(user_input)

        # Place the assistant's response in the chat with a spinner
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # We'll always pass the 512px-high masked image to whichever model is selected
                #image_to_send = st.session_state["masked_image"]
                image_to_send = original_image

                if st.session_state["model_choice"] in ["gemini-1.5-pro", "gemini-2.0-flash"]:
                    response_text = gemini.request_gemini_chat(
                        model_name=st.session_state["model_choice"],
                        user_message=user_input,
                        pil_image=image_to_send,
                        conversation_history=st.session_state["conversation_history"],
                    )
                else:
                    # GPT-based model ("gpt-4o")
                    response_text = gpt.request_gpt_chat(
                        model_name=st.session_state["model_choice"],
                        user_message=user_input,
                        pil_image=image_to_send,
                        conversation_history=st.session_state["conversation_history"],
                    )

            # Once the model returns a response, we display it in the same assistant message
            st.write(response_text)
        # Save user & assistant messages
        st.session_state["conversation_history"].append({"role": "user", "content": user_input})
        st.session_state["conversation_history"].append({"role": "assistant", "content": response_text})
        print(st.session_state["conversation_history"])

        st.stop()

if __name__ == "__main__":
    main()
