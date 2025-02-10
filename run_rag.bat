@echo off

REM 检查是否安装了必要的包
pip install -r requirements.txt

REM 运行文件问答助手
streamlit run rag.py