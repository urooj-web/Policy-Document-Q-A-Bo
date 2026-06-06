import os
import gradio as gr
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory.buffer import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq

# ─── Global state ─────────────────────────────────────────────────────────────
conversation = None

# ─── Core functions ───────────────────────────────────────────────────────────
def get_pdf_text(pdf_files):
    text = ""
    for pdf in pdf_files:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

def get_text_chunks(raw_text):
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return splitter.split_text(raw_text)

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)

def get_conversation_chain(vectorstore):
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY secret not set. Add it in Space Settings → Secrets.")
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0.7,
        request_timeout=30,
        api_key=groq_api_key
    )
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )

# ─── Gradio handlers ──────────────────────────────────────────────────────────
def process_pdfs(pdf_files):
    global conversation
    if not pdf_files:
        return "⚠️ Please upload at least one PDF.", gr.update(interactive=False)
    try:
        raw_text = get_pdf_text(pdf_files)
        if not raw_text.strip():
            return "❌ No text found. PDF may be scanned/image-based.", gr.update(interactive=False)
        chunks = get_text_chunks(raw_text)
        vectorstore = get_vectorstore(chunks)
        conversation = get_conversation_chain(vectorstore)
        return f"✅ Processed {len(pdf_files)} PDF(s) — {len(chunks)} chunks. Ask your question!", gr.update(interactive=True)
    except Exception as e:
        return f"❌ Error: {str(e)}", gr.update(interactive=False)

def chat(question, history):
    global conversation
    if not question.strip():
        return history, ""
    if conversation is None:
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": "⚠️ Please upload and process a PDF first."})
        return history, ""
    try:
        response = conversation({"question": question})
        answer = response["answer"]
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})
        return history, ""
    except Exception as e:
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": f"❌ Error: {str(e)}"})
        return history, ""

def clear_chat():
    global conversation
    if conversation:
        conversation.memory.clear()
    return [], ""

# ─── UI ───────────────────────────────────────────────────────────────────────
with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="blue", secondary_hue="indigo"),
    title="Policy Document Q&A Bot"
) as demo:

    gr.Markdown("""
    # 📄 Policy Document Q&A Bot
    Upload any policy PDF and ask questions in natural language using AI.
    > Powered by **LangChain · FAISS · HuggingFace Embeddings · Groq LLaMA 3.3**
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📂 Step 1 — Upload PDF(s)")
            pdf_input = gr.File(
                label="Upload Policy Document(s)",
                file_types=[".pdf"],
                file_count="multiple"
            )
            process_btn = gr.Button("⚙️ Process Documents", variant="primary", size="lg")
            status_box = gr.Textbox(
                label="Status",
                value="Upload a PDF and click Process to begin.",
                interactive=False,
                lines=3
            )

        with gr.Column(scale=2):
            gr.Markdown("### 💬 Step 2 — Ask Questions")
            chatbot = gr.Chatbot(
                label="Policy Assistant",
                height=420,
                type="messages"
            )
            with gr.Row():
                question_input = gr.Textbox(
                    label="Your Question",
                    placeholder="e.g. What is the National Education Policy about?",
                    lines=2,
                    interactive=False,
                    scale=4
                )
                submit_btn = gr.Button("Send 🚀", variant="primary", scale=1)
            clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary")

    

    # Events
    process_btn.click(fn=process_pdfs, inputs=[pdf_input], outputs=[status_box, question_input])
    submit_btn.click(fn=chat, inputs=[question_input, chatbot], outputs=[chatbot, question_input])
    question_input.submit(fn=chat, inputs=[question_input, chatbot], outputs=[chatbot, question_input])
    clear_btn.click(fn=clear_chat, outputs=[chatbot, question_input])

if __name__ == "__main__":
    demo.launch()
