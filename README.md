# RAPBench

RAPBench is a lightweight application designed to benchmark the planning capabilities of large multimodal models (LMMs), focusing on **ChatGPT-4o** for **robot action planning (RAP)** tasks in domestic and office environments. The interface supports **multi-image inputs** and **turn-based interactions**, enabling iterative plan refinement.

Built with **Streamlit**, this application allows real-time testing and export of performance data, including robot action sequences and interaction metrics.

---

## 🚀 Features

- 📷 **Multimodal Input:** Upload multiple room images as visual context.
- 💬 **Turn-Based Planning:** Issue natural language instructions and refine plans through follow-up interactions.
- 🧠 **Plan Visualization:** Display structured robot action plans in tabular format.
- 📈 **Metrics Logging:** Track questions, response length, latency, and token usage per iteration.
- 📤 **Export & Comparison:** Compare RAPs and logs between prompt iterations.

## Requirements

- Python 3.8+
- [Streamlit](https://streamlit.io/)
- Additional dependencies as listed in `requirements.txt`

## Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/yourusername/RAPBench.git
    cd RAPBench
    ```

2. **Set up your virtual environment and install dependencies:**

    ```bash
    python3 -m venv env
    source env/bin/activate  # For Windows use: env\Scripts\activate
    pip install -r requirements.txt
    ```

## Usage

Start the Streamlit interface by running:


If you want to use the latest app made for the Robot Action Plan Benchmarking:
```bash
cd rap_app
streamlit run rap_app_standalone.py
```

Else if you want to use the previous version used for the question and answer benchmark with
gemini object detection: 
```bash
cd app
streamlit run app.py
```
