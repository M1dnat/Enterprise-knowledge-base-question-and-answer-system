# 📚 企业知识库问答系统

基于 LangChain 和 Streamlit 构建的企业知识库智能问答系统，支持文档管理、向量检索、RAG 问答、答案溯源等多种实用功能。

## ✨ 功能特性

| 功能 | 描述 | 示例 |
|------|------|------|
| 📄 文档批量导入 | 支持批量导入TXT文档，自动处理 | 上传产品手册、公司制度、FAQ |
| ✂️ 智能文本分块 | 自动将长文档切分为合适大小的文本块 | chunk_size=500, overlap=50 |
| 🔢 向量化存储 | 使用BGE-M3模型将文本转为向量 | 存储到FAISS向量数据库 |
| 🔍 语义检索 | 基于向量相似度搜索相关内容 | "公司主要产品有哪些？" |
| 🤖 RAG问答 | 结合检索结果和大模型生成答案 | 综合多个文档生成准确回答 |
| 📚 答案溯源 | 显示答案来源文档和具体内容片段 | 点击查看参考文档详情 |
| 📊 查询日志 | 自动记录所有问答历史 | 导出JSON格式日志文件 |
| 💾 索引持久化 | 向量索引保存到本地，支持快速加载 | 下次启动无需重新构建 |

## 🚀 快速开始

### 📋 环境要求

- Python 3.11+
- SiliconFlow API Key

### 📥 安装步骤

1. 克隆仓库
   git clone https://github.com/Mldnat/Enterprise-knowledge-base-question-and-answer-system.git
   cd Enterprise-knowledge-base-question-and-answer-system

2. 安装依赖
   pip install streamlit langchain langchain-community langchain-openai faiss-cpu python-dotenv

3. 配置 API Key
   echo "API_KEY=your_api_key_here" > .env

4. 运行应用
   streamlit run app.py

5. 访问界面
   打开浏览器访问：http://localhost:8501

## 📁 项目结构

Enterprise-knowledge-base-question-and-answer-system/
├── 📄 app.py                    # 主程序文件
├── 🔐 .env                      # 配置文件
├── 📦 requirements.txt          # 依赖列表
├── 📁 knowledge_docs/           # 文档存储目录
└── 📁 knowledge_index/          # 向量索引目录

## 🌐 公网访问

cpolar http 8501

## 📄 License

MIT
