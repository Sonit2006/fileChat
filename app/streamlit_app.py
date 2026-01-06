# streamlit_app.py

import streamlit as st
import cloud_engine as engine
import os

st.set_page_config(
    page_title="Local AI Chatbot",
    layout="wide"
)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_chain" not in st.session_state:
    with st.spinner("Loading engine..."):
        db = engine.load_db()
        if db is None:
            st.session_state.rag_chain = None
            st.error("Failed to load document database.")
        else:
            st.session_state.rag_chain = engine.create_rag_chain(db)
            st.session_state.messages.append({
                "role": "assistant",
                "content": "Ask me anything about your documents."
            })

st.title("Local AI Chatbot")
st.markdown("---")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)


def handle_voice_input():
    try:
        import speech_recognition as sr
    except ImportError:
        st.error("speech_recognition is not installed.")
        return None

    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening...")
        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
            text = recognizer.recognize_google(audio)
            st.success(f"Heard: {text}")
            return text
        except sr.UnknownValueError:
            st.warning("Could not understand audio.")
        except sr.RequestError as e:
            st.error(f"Speech recognition error: {e}")
        except Exception as e:
            st.error(str(e))

    return None


def process_query(query: str):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        with st.spinner("Thinking..."):
            try:
                result = st.session_state.rag_chain.invoke({"input": query})
                answer = result.get("answer", "No answer found.")
                context = result.get("context", [])

                sources = {
                    doc.metadata.get("source", "Unknown")
                    for doc in context
                }

                placeholder.markdown(answer)

                if sources:
                    st.markdown("**Sources:**")
                    for i, path in enumerate(sorted(sources)):
                        if not os.path.exists(path):
                            st.warning(f"Missing file: {path}")
                            continue

                        name = os.path.basename(path)
                        try:
                            with open(path, "rb") as f:
                                st.download_button(
                                    label=f"Open: {name}",
                                    data=f,
                                    file_name=name,
                                    key=f"open_{i}"
                                )
                        except Exception as e:
                            st.error(f"Failed to open {name}: {e}")

                history_text = answer
                if sources:
                    history_text += (
                        "\n\n**Sources:**\n" +
                        "\n".join(os.path.basename(s) for s in sources)
                    )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": history_text
                })

            except Exception as e:
                placeholder.error(f"Error: {e}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"Error: {e}"
                })


col_main, col_side = st.columns([4, 1])

with col_side:
    if st.button("Speak", use_container_width=True):
        if st.session_state.rag_chain:
            spoken = handle_voice_input()
            if spoken:
                process_query(spoken)

with col_main:
    prompt = st.chat_input("Ask about your documents")
    if prompt and st.session_state.rag_chain:
        process_query(prompt)
