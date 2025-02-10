@echo off

REM 检查是否安装了必要的包
pip install -r requirements.txt

REM 运行聊天助手
streamlit run chat.py