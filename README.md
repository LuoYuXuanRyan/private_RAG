# AI文件问答与对话助手

这是一个基于Streamlit和LangChain开发的AI助手系统，包含两个主要功能模块：文件问答助手和AI对话助手。

## 功能特点

### 文件问答助手
- 支持多种文件格式的上传和解析（PDF、Word、Excel、PowerPoint）
- 使用RAG（检索增强生成）技术实现精准的文件内容问答
- 支持上下文关联的多轮对话
- 实时流式回答输出

### AI对话助手
- 基于大语言模型的智能对话系统
- 友好的用户界面
- 支持连续对话
- 实时流式回答输出

## 安装要求

1. 确保已安装Python 3.8+
2. 安装Ollama并运行以下模型：
   - deepseek-r1:7b
   - nomic-embed-text

3. 安装依赖包：
```bash
pip install -r requirements.txt
```

## 快速开始

### 运行文件问答助手
```bash
streamlit run rag.py
```
或直接双击运行 `run_rag.bat`

### 运行AI对话助手
```bash
streamlit run chat.py
```
或直接双击运行 `run_chat.bat`

## 使用说明

### 文件问答助手
1. 点击"上传文件"按钮上传需要分析的文档
2. 等待文件处理完成
3. 在对话框中输入问题
4. 系统会基于文档内容提供答案
5. 可以使用"清除聊天历史"按钮重置对话

### AI对话助手
1. 直接在对话框中输入问题
2. 系统会实时生成回答
3. 可以使用"清除聊天历史"按钮重置对话

## 技术栈
- Streamlit：Web界面框架
- LangChain：大语言模型应用框架
- ChromaDB：向量数据库
- Ollama：本地大语言模型部署
