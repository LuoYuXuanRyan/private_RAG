import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

# 设置页面标题
st.set_page_config(page_title="AI对话助手", layout="wide")
st.title("AI对话助手")

# 初始化会话状态
if "messages" not in st.session_state:
    st.session_state.messages = []

# 设置模型和提示模板
model = ChatOllama(model="deepseek-r1:7b")

SYSTEM_TEMPLATE = """
Role: AI对话助手

Profile
Language: 中文
Description: 我是一个友好的AI助手，可以帮助回答各种问题，提供信息和建议。

Rules
1. 保持友好专业的对话态度
2. 如果不确定，要明确告知
3. 保持客观，不添加主观意见
4. 回答要简洁清晰
"""

CHAT_TEMPLATE = """
{system_message}

用户问题: {question}

请提供清晰、准确的回答。
"""

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_TEMPLATE),
    ("user", CHAT_TEMPLATE)
])

# 显示聊天历史
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 聊天输入
if question := st.chat_input("请输入您的问题"):
    # 添加用户问题到聊天历史
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # 创建对话链
    chat_chain = (
        {
            "question": lambda x: x,
            "system_message": lambda _: SYSTEM_TEMPLATE
        }
        | chat_prompt
        | model
        | StrOutputParser()
    )

    # 显示助手回答
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # 使用流式输出
        for chunk in chat_chain.stream(question):
            full_response += chunk
            message_placeholder.markdown(full_response + "▌")
        
        # 完成后显示完整消息，去掉光标
        message_placeholder.markdown(full_response)
        # 将完整回答添加到聊天历史
        st.session_state.messages.append({"role": "assistant", "content": full_response})

# 清除聊天历史的按钮
if st.button("清除聊天历史"):
    st.session_state.messages = []
    st.rerun() 