import streamlit as st
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import json
import re
import io

from requests_ai import generate_masked_image, request_gpt4o

def extract_json_snippet(text: str) -> str:
    """Extract the first JSON array from the assistant’s reply."""
    m = re.search(r"```json\s*(\[\s*{.*?}\s*\])\s*```", text, re.DOTALL)
    if m:
        return m.group(1)
    m = re.search(r"(\[\s*{.*?}\s*\])", text, re.DOTALL)
    return m.group(1) if m else "[]"

def strip_rap_label_and_json(full_text: str) -> str:
    """
    Remove all fenced code blocks (```...```), drop any 'RAP:' lines,
    and return only the narrative text.
    """
    # 1) Strip out any ```…``` block entirely
    without_fences = re.sub(r"```[\s\S]*?```", "", full_text)
    # 2) Split into lines and remove any line starting with "RAP:"
    lines = [
        line for line in without_fences.splitlines()
        if not line.strip().startswith("RAP:")
    ]
    # 3) Rejoin and trim
    return "\n".join(lines).strip()

def main():
    st.set_page_config(layout="wide")
    st.title("RAP-Bench: Interactive Robot Action Plan Refinement")

    # ─── Session State ──────────────────────────────────────────
    st.session_state.setdefault("rap_versions", [])
    st.session_state.setdefault("rap_changes", [])
    st.session_state.setdefault("masked_image", None)
    st.session_state.setdefault("iteration_data", [])

    # ─── 1) Image Uploader & Masking ────────────────────────────
    uploaded = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])
    if not uploaded:
        # Reset if cleared
        st.session_state["rap_versions"].clear()
        st.session_state["rap_changes"].clear()
        st.session_state["masked_image"] = None
        st.session_state["iteration_data"].clear()
        st.info("Please upload an image to begin.")
        return

    original = Image.open(uploaded)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(original, width=500)
    with col2:
        st.subheader("Masked Image (512px high)")
        if st.session_state["masked_image"] is None:
            with st.spinner("Running Gemini 2.0…"):
                full_masked = generate_masked_image(
                    original, prompt="Detect all objects in the image."
                )
                w, h = full_masked.size
                if h > 512:
                    scale = 512 / h
                    full_masked = full_masked.resize(
                        (int(w * scale), 512),
                        Image.Resampling.LANCZOS
                    )
            st.session_state["masked_image"] = full_masked
        st.image(st.session_state["masked_image"], width=500)

    st.write("---")

    # ─── 2) Chat & RAP Columns ──────────────────────────────────
    left, right = st.columns([2,3])

# 2a) CHAT PANEL
    with left:
        st.subheader("Chat")
        user_input = st.chat_input("Type something like 'Make egg scramble'")

        if user_input:
            # Echo user message immediately
            st.chat_message("user").write(user_input)

            # Build context from past iterations
            context = []
            for entry in st.session_state["iteration_data"]:
                context.append({"role": "user",      "content": entry["User"]})
                context.append({"role": "assistant", "content": entry["assistant_full"]})

            # Placeholder + spinner for assistant
            with st.chat_message("assistant"):
                placeholder = st.empty()
                with st.spinner("Thinking…"):
                    full_reply = request_gpt4o(
                        user_message=user_input,
                        conversation_history=context,
                        pil_image=st.session_state["masked_image"]
                    )

                narrative = strip_rap_label_and_json(full_reply)
                placeholder.write(narrative)

                print("Full reply:", full_reply)
                print("Narrative:", narrative)

            # Parse and store new RAP table
            snippet = extract_json_snippet(full_reply)
            try:
                rap_list = json.loads(snippet)
                df = pd.DataFrame(rap_list)
            except:
                df = pd.DataFrame()
            st.session_state["rap_versions"].append(df)

            # Compute and store change log entries
            if len(st.session_state["rap_versions"]) > 1:
                prev = st.session_state["rap_versions"][-2]
                cur  = st.session_state["rap_versions"][-1]
                new_j = cur.apply(lambda r: r.to_json(), axis=1)
                old_j = prev.apply(lambda r: r.to_json(), axis=1)
                added = cur.loc[~new_j.isin(old_j)]
                iteration = len(st.session_state["rap_versions"])
                for _, row in added.iterrows():
                    st.session_state["rap_changes"].append(
                        {"Iteration": iteration, **row.to_dict()}
                    )

            # Record this iteration’s data
            st.session_state["iteration_data"].append({
                "Iteration": len(st.session_state["rap_versions"]),
                "User": user_input,
                "Assistant": narrative,
                "assistant_full": full_reply
            })

    # 2b) RAP, Change Log & Iteration History
    with right:
        st.subheader("Robot Action Plan")

        # RAP Table
        if st.session_state["rap_versions"]:
            latest = st.session_state["rap_versions"][-1]
            if len(st.session_state["rap_versions"]) > 1:
                prev = st.session_state["rap_versions"][-2]
                nj = latest.apply(lambda r: r.to_json(), axis=1)
                oj = prev.apply(lambda r: r.to_json(), axis=1)
                mask = ~nj.isin(oj)
                def highlight_text(row):
                    # If this row is “new”, color all its text yellow; otherwise leave default
                    return ["color: yellow"] * len(row) if mask.loc[row.name] else [""] * len(row)
                st.dataframe(latest.style.apply(highlight_text, axis=1), use_container_width=True)
            else:
                st.dataframe(latest, use_container_width=True)
        else:
            st.info("Your RAP will appear here after the first response.")

        # Change Log expander
        with st.expander("Change Log", expanded=False):
            if st.session_state["rap_changes"]:
                st.table(pd.DataFrame(st.session_state["rap_changes"]))
            else:
                st.write("No changes logged yet.")

        # Iteration History expander
        with st.expander("Iteration History", expanded=False):
            if st.session_state["iteration_data"]:
                hist_df = pd.DataFrame(st.session_state["iteration_data"])[
                    ["Iteration", "User", "Assistant"]
                ]
                st.table(hist_df)
            else:
                st.write("No iterations yet.")

    # ─── 3) Per-Iteration Metrics (full-width under both columns) ─────────────────────────────
    if st.session_state["rap_versions"]:
        st.write("---")
        st.subheader("Per-Iteration Metrics")

        # iteration indices
        iters = list(range(1, len(st.session_state["rap_versions"]) + 1))

        # metric series
        actions      = [len(df) for df in st.session_state["rap_versions"]]
        questions    = [entry["assistant_full"].count("?") for entry in st.session_state["iteration_data"]]
        avg_resp_len = [len(entry["assistant_full"].split()) for entry in st.session_state["iteration_data"]]
        rap_cols     = [len(df.columns) for df in st.session_state["rap_versions"]]

        # four side-by-side panels
        chart_cols = st.columns(4)
        specs = [
            ("# Actions",         actions,      "# Actions"),
            ("# Questions",       questions,    "# Questions"),
            ("Avg. Resp. Length", avg_resp_len, "Words"),
            ("# RAP Columns",     rap_cols,     "# Columns"),
        ]

        for (title, data, ylabel), col in zip(specs, chart_cols):
            fig, ax = plt.subplots(figsize=(3, 2), dpi=100)
            ax.plot(iters, data, marker="o", linewidth=4)
            ax.set_title(title, pad=10)
            ax.set_xlabel("Iter")
            ax.set_ylabel(ylabel)

            # force x ticks to the exact integer iterations
            ax.set_xticks(iters)

            # adjust y-limits: stretch Avg. Resp. Length chart higher
            if title == "Avg. Resp. Length":
                top = max(data) * 1.2 if data else 1
                ax.set_ylim(0, top)
            else:
                ax.set_ylim(0, max(data) + 1 if data else 1)

            ax.grid(alpha=0.25)
            col.pyplot(fig, use_container_width=True, clear_figure=True)

    # ─── 4) Export All Data (Single‐sheet) ────────────────────────────────────
    st.write("---")
    if st.button("📥 Export all data to Excel (single sheet)"):
        rows = []
        for i, data in enumerate(st.session_state["iteration_data"], start=1):
            user_msg      = data["User"]
            assistant_narr= data["Assistant"]
            assistant_full= data["assistant_full"]

            # flatten JSON RAP
            snippet = extract_json_snippet(assistant_full)
            rap_str = snippet.replace("\n", " ").strip()

            # collect only those change‐dicts for this iteration
            changes = [
                {k: v for k, v in entry.items() if k != "Iteration"}
                for entry in st.session_state["rap_changes"]
                if entry.get("Iteration") == i
            ]
            # join them into one cell
            change_str = "; ".join(str(d) for d in changes) if changes else ""

            # metrics
            df      = st.session_state["rap_versions"][i-1]
            num_actions   = len(df)
            num_questions = assistant_full.count("?")
            avg_resp_len  = len(assistant_full.split())
            num_cols      = len(df.columns)

            rows.append({
                "Iteration":         i,
                "User message":      user_msg,
                "Assistant narrative": assistant_narr,
                "RAP JSON":          rap_str,
                "Change log":        change_str,
                "# Actions":         num_actions,
                "# Questions":       num_questions,
                "Avg resp. length":  avg_resp_len,
                "# RAP columns":     num_cols,
            })

        export_df = pd.DataFrame(rows)

        # write to in‐memory Excel
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
            export_df.to_excel(writer, sheet_name="RAP_Bench", index=False)
        buffer.seek(0)

        st.download_button(
            "Download all data as Excel",
            data=buffer,
            file_name="rap_bench_export.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
if __name__ == "__main__":
    main()
