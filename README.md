# GeminiQBench

GeminiQBench is a simple application designed to benchmark various LMMS models on computer vision tasks. It integrates Gemini 2.0 object detection with an interactive, multi-turn conversational interface built with Streamlit. The system dynamically refines its questions based on user input to accurately identify a specific object in an image.

## Features

- **Gemini 2.0 Object Detection:** Processes input images to generate masked outputs and bounding boxes.
- **Multi-Turn Conversational Interface:** Enables dynamic questioning to refine object detection based on user responses.
- **Benchmarking Framework:** Compares the performance of different LMMS models in interactive visual tasks.

## Requirements

- Python 3.8+
- [Streamlit](https://streamlit.io/)
- Additional dependencies as listed in `requirements.txt`

## Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/yourusername/GeminiQBench.git
    cd GeminiQBench
    ```

2. **Set up your virtual environment and install dependencies:**

    ```bash
    python3 -m venv env
    source env/bin/activate  # For Windows use: env\Scripts\activate
    pip install -r requirements.txt
    ```

## Usage

Start the Streamlit interface by running:

```bash
streamlit run app.py
