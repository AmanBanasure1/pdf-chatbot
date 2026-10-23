
import streamlit as st

from rag_engine import (
    extract_pdf,
    create_chunks,
    build_index,
    retrieve,
    generate_answer
)

st.set_page_config(
    page_title="RAG PDF Chat",
    page_icon="📚",
    layout="centered"
)

st.title("📚 RAG PDF Chat")
st.caption("Apne PDF se questions poochho — powered by RAG")

st.info(
    "Free local retrieval available hai. "
    "AI-generated answers ke liye optional Groq API key use karo."
)

with st.sidebar:
    st.header("Settings")
    api_key = st.text_input(
        "Groq API key (optional)",
        type="password",
        help="Key ko public code ya GitHub mein mat daalo."
    )

    top_k = st.slider(
        "Relevant passages",
        min_value=1,
        max_value=6,
        value=4
    )

    st.caption(
        "API key sirf current session ke input mein use hoti hai."
    )

uploaded_file = st.file_uploader(
    "PDF upload karo",
    type=["pdf"]
)

if uploaded_file is not None:
    st.write(f"**Selected file:** {uploaded_file.name}")

    if st.button("Process PDF", type="primary"):
        try:
            with st.spinner("PDF read aur index ho raha hai..."):
                pages = extract_pdf(uploaded_file)
                chunks = create_chunks(pages)
                vectorizer, matrix = build_index(chunks)

                st.session_state["chunks"] = chunks
                st.session_state["vectorizer"] = vectorizer
                st.session_state["matrix"] = matrix
                st.session_state["doc_name"] = uploaded_file.name
                st.session_state["messages"] = []

            st.success(
                f"PDF ready! {len(pages)} readable pages, "
                f"{len(chunks)} text chunks."
            )

        except Exception as error:
            st.error(f"PDF process nahi hui: {error}")

if "chunks" in st.session_state:
    st.divider()
    st.subheader("💬 Ask your PDF")

    st.caption(
        f"Loaded document: {st.session_state['doc_name']}"
    )

    for message in st.session_state.get("messages", []):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message.get("sources"):
                with st.expander("View retrieved sources"):
                    for source in message["sources"]:
                        st.markdown(
                            f"**Page {source['page']}** "
                            f"(score: {source['score']:.3f})"
                        )
                        st.write(source["text"])

    question = st.chat_input(
        "PDF ke baare mein question poochho..."
    )

    if question:
        st.session_state["messages"].append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Relevant passages search ho rahe hain..."):
                sources = retrieve(
                    question,
                    st.session_state["chunks"],
                    st.session_state["vectorizer"],
                    st.session_state["matrix"],
                    top_k=top_k
                )

                answer, mode = generate_answer(
                    question,
                    sources,
                    api_key=api_key.strip() or None
                )

            st.markdown(answer)
            st.caption(f"Mode: {mode}")

            if sources:
                with st.expander("View retrieved sources"):
                    for source in sources:
                        st.markdown(
                            f"**Page {source['page']}** "
                            f"(score: {source['score']:.3f})"
                        )
                        st.write(source["text"])

        st.session_state["messages"].append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })

elif uploaded_file is None:
    st.markdown(
        "1. PDF upload karo.\n"
        "2. **Process PDF** par click karo.\n"
        "3. Apna question poochho."
    )
