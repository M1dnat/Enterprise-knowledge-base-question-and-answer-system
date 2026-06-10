@echo off
echo ========================================
echo 正在安装企业知识库问答系统依赖
echo ========================================

echo 升级pip...
python -m pip install --upgrade pip

echo 安装依赖包...
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple ^
    streamlit==1.35.0 ^
    langchain==0.1.0 ^
    langchain-community==0.0.10 ^
    langchain-text-splitters==0.0.1 ^
    langchain-openai==0.0.5 ^
    faiss-cpu==1.7.4 ^
    python-dotenv==1.0.0

echo ========================================
echo 安装完成！
echo ========================================
echo 按任意键退出...
pause