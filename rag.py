import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain.docstore.document import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tempfile
import os
import shutil
import sqlite3
from chromadb.config import Settings
import docx2txt

# 设置页面
st.set_page_config(page_title="文件问答助手", layout="wide")
st.title("文件问答助手")

# 定义持久化路径
PERSIST_DIR = "./chroma_db"
DB_PATH = "chat_history.db"

# 确保持久化目录存在
if not os.path.exists(PERSIST_DIR):
    os.makedirs(PERSIST_DIR)

# 确保 SQLite 数据库初始化
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT,
            content TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# 读取历史聊天记录
def load_chat_history():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT role, content FROM messages")
    messages = [{"role": row[0], "content": row[1]} for row in cursor.fetchall()]
    conn.close()
    return messages

# 存储聊天记录
def save_chat_message(role, content):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO messages (role, content) VALUES (?, ?)", (role, content))
    conn.commit()
    conn.close()

# 清空聊天记录
def clear_chat_history():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM messages")
    conn.commit()
    conn.close()

# Chroma 配置
CHROMA_SETTINGS = Settings(
    allow_reset=True,
    is_persistent=True,
    persist_directory=PERSIST_DIR
)

# 初始化会话状态
if "messages" not in st.session_state:
    st.session_state.messages = load_chat_history()
if "vectorstore" not in st.session_state:
    try:
        local_embeddings = OllamaEmbeddings(model="nomic-embed-text")
        st.session_state.vectorstore = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=local_embeddings,
            collection_name="document_collection",
            client_settings=CHROMA_SETTINGS,
        )
        if st.session_state.vectorstore._collection.count() > 0:
            st.success("已加载现有知识库!")
    except Exception as e:
        st.error(f"加载知识库时发生错误: {str(e)}")
        st.session_state.vectorstore = None

# 文件上传
uploaded_files = st.file_uploader("上传文件", type=['pdf', 'docx'], accept_multiple_files=True)

if uploaded_files:
    all_docs = []
    for uploaded_file in uploaded_files:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        try:
            if file_extension == 'pdf':
                loader = PDFPlumberLoader(tmp_file_path)
                docs = loader.load()
            elif file_extension == 'docx':
                try:
                    text = docx2txt.process(tmp_file_path)
                    docs = [Document(page_content=text, metadata={"source": "docx", "file": uploaded_file.name})]
                except Exception as e:
                    st.error(f"处理 .docx 文件时发生错误: {str(e)}")
                    docs = []
            
            all_docs.extend(docs)
            os.unlink(tmp_file_path)
            st.success(f"{uploaded_file.name} 已成功上传并处理!")
            
        except Exception as e:
            st.error(f"处理文件 {uploaded_file.name} 时发生错误: {str(e)}")
            os.unlink(tmp_file_path)

    if all_docs:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        all_splits = text_splitter.split_documents(all_docs)
        
        local_embeddings = OllamaEmbeddings(model="nomic-embed-text")
        
        if st.session_state.vectorstore is None:
            st.session_state.vectorstore = Chroma.from_documents(
                documents=all_splits, 
                embedding=local_embeddings,
                collection_name="document_collection",
                persist_directory=PERSIST_DIR,
                client_settings=CHROMA_SETTINGS
            )
        else:
            st.session_state.vectorstore.add_documents(all_splits)
        
        # st.session_state.vectorstore.persist()
        st.success("知识库已更新并保存!")

# 设置模型和提示模板
model = ChatOllama(model="deepseek-r1:7b")

rag_prompt = ChatPromptTemplate.from_messages([
    ("system", "Role: 文件信息处理专家\n\n请使用以下检索到的上下文来回答问题。\n\n<context>\n{context}\n</context>\n\n问题: {question}\n"),
    ("user", "{question}")
])

# 格式化文档
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 显示历史消息
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 处理用户输入
if question := st.chat_input("请输入您的问题"):
    st.session_state.messages.append({"role": "user", "content": question})
    save_chat_message("user", question)

    with st.chat_message("user"):
        st.markdown(question)

    if st.session_state.vectorstore is None:
        with st.chat_message("assistant"):
            st.markdown("请先上传文件!")
        st.session_state.messages.append({"role": "assistant", "content": "请先上传文件!"})
        save_chat_message("assistant", "请先上传文件!")
    else:
        retriever = st.session_state.vectorstore.as_retriever()
        qa_chain = (
            {
                "context": retriever | format_docs, 
                "question": lambda x: x
            }
            | rag_prompt
            | model
            | StrOutputParser()
        )

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            for chunk in qa_chain.stream(question):
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")

            message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            save_chat_message("assistant", full_response)

# 清除聊天记录
if st.button("清除聊天历史"):
    clear_chat_history()
    st.session_state.messages = []
    st.rerun()

# 清除知识库
if st.button("清除知识库"):
    try:
        if st.session_state.vectorstore is not None:
            st.session_state.vectorstore.delete_collection()
        if os.path.exists(PERSIST_DIR):
            shutil.rmtree(PERSIST_DIR)
            os.makedirs(PERSIST_DIR)
        st.session_state.vectorstore = None
        st.success("知识库已清除!")
        st.rerun()
    except Exception as e:
        st.error(f"清除知识库时发生错误: {str(e)}")