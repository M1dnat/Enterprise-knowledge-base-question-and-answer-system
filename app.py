"""
企业知识库问答系统 - 单文件版本
使用 Streamlit + LangChain + SiliconFlow API
"""

import os
import json
from datetime import datetime
from typing import List, Dict

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter  # 注意是 text_splitters
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 加载环境变量
load_dotenv()
API_KEY = os.getenv("API_KEY")

if not API_KEY:
    st.error("❌ 请在 .env 文件中设置 API_KEY")
    st.stop()


class KnowledgeBase:
    """企业知识库系统"""
    
    def __init__(self, docs_directory: str = "./knowledge_docs", index_directory: str = "./knowledge_index"):
        self.docs_directory = docs_directory
        self.index_directory = index_directory
        
        # 创建目录
        os.makedirs(self.docs_directory, exist_ok=True)
        
        # 初始化 Embedding 模型
        self.embeddings = OpenAIEmbeddings(
            base_url="https://api.siliconflow.cn/v1",
            api_key=API_KEY,
            model="BAAI/bge-m3"
        )
        
        # 文本分块器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
        )
        
        # 向量数据库
        self.vectorstore = None
        
        # 初始化 LLM
        self.model = init_chat_model(
            "Qwen/Qwen3-8B",
            model_provider="openai",
            base_url="https://api.siliconflow.cn/v1",
            api_key=API_KEY,
            temperature=0.0
        )
        
        # 查询历史
        self.query_log = []
    
    def build_index(self) -> bool:
        """构建索引"""
        st.info("📁 正在加载文档...")
        
        loader = DirectoryLoader(
            self.docs_directory,
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
            show_progress=True
        )
        
        try:
            docs = loader.load()
            st.info(f"✅ 成功加载 {len(docs)} 个文档")
        except Exception as e:
            st.error(f"❌ 加载文档失败: {e}")
            return False
        
        if not docs:
            st.warning("⚠️ 没有找到文档，请确保文档目录中有 .txt 文件")
            return False
        
        st.info(f"📦 正在分块...")
        split_docs = self.text_splitter.split_documents(docs)
        st.info(f"✅ 分块完成，共 {len(split_docs)} 个文本块")
        
        st.info(f"🔢 正在向量化并建立索引...")
        self.vectorstore = FAISS.from_documents(split_docs, self.embeddings)
        
        os.makedirs(self.index_directory, exist_ok=True)
        self.vectorstore.save_local(self.index_directory)
        st.success(f"✅ 索引已保存到 {self.index_directory}")
        
        return True
    
    def load_index(self) -> bool:
        """加载已有索引"""
        try:
            self.vectorstore = FAISS.load_local(
                self.index_directory,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            return True
        except Exception as e:
            return False
    
    def query(self, question: str, k: int = 3) -> Dict:
        """查询知识库"""
        if not self.vectorstore:
            return {"error": "知识库未初始化"}
        
        # 1. 检索相关文档
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )
        docs = retriever.invoke(question)
        
        # 2. 构建提示词
        prompt = ChatPromptTemplate.from_template("""你是企业知识库问答助手。

请基于以下文档片段回答问题。

文档片段:
{context}

用户问题: {question}

回答要求:
1. 仅基于提供的文档内容回答
2. 如果文档中没有相关信息，明确告知用户
3. 回答要准确、完整、易懂
4. 如果答案来自多个文档片段，请综合回答

答案: """)
        
        # 3. 格式化文档
        def format_docs(docs):
            return "\n\n---\n\n".join([
                f"【文档 {i+1}】\n来源: {doc.metadata.get('source', '未知')}\n内容: {doc.page_content}"
                for i, doc in enumerate(docs)
            ])
        
        # 4. 构建 RAG Chain
        rag_chain = (
            {"context": lambda x: format_docs(docs), "question": RunnablePassthrough()}
            | prompt
            | self.model
            | StrOutputParser()
        )
        
        # 5. 生成答案
        answer = rag_chain.invoke(question)
        
        # 6. 记录查询
        self.query_log.append({
            "question": question,
            "answer": answer,
            "sources": [doc.metadata.get('source', '未知') for doc in docs],
            "timestamp": datetime.now().isoformat()
        })
        
        return {
            "question": question,
            "answer": answer,
            "sources": docs,
            "source_count": len(docs)
        }
    
    def export_logs(self, filename: str = "query_log.json"):
        """导出查询日志"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.query_log, f, ensure_ascii=False, indent=2)
        st.success(f"✅ 查询日志已导出到 {filename}")


def create_sample_docs(directory: str = "./knowledge_docs"):
    """创建示例文档"""
    os.makedirs(directory, exist_ok=True)
    
    docs = {
        "company_intro.txt": """公司简介

我们是一家专注于人工智能技术研发的科技公司，成立于2020年。

公司使命：让AI技术普惠每个人。

主要产品：
1. AI开发平台 - CloudAI Platform
2. 智能客服系统 - SmartCS
3. 知识库问答系统 - KnowledgeBase

公司规模：员工200人，其中研发人员占70%。

公司地址：北京市海淀区中关村软件园。
""",
        "product_ai_platform.txt": """AI开发平台

产品名称：CloudAI Platform

功能特性：
1. 模型训练：支持多种深度学习框架
2. 模型部署：一键部署到云端
3. API接口：提供RESTful API
4. 监控告警：实时监控模型性能

技术优势：
- 支持分布式训练
- 自动调参
- 模型版本管理
- A/B测试

定价方案：
- 免费版：每月1000次API调用
- 专业版：每月￥999，10万次调用
- 企业版：定制化，联系销售
""",
        "customer_service.txt": """智能客服系统

产品名称：SmartCS

核心功能：
1. 多轮对话：支持复杂的多轮对话
2. 知识库：集成企业知识库
3. 工单系统：自动创建工单
4. 数据分析：客服数据可视化

适用场景：
- 电商客服
- 银行客服
- 政府服务热线

接入方式：
- 网页插件
- 微信公众号
- 小程序
- APP SDK

客户案例：
某电商平台接入后，客服效率提升60%，客户满意度提高35%。
""",
        "hr_benefits.txt": """员工福利体系

薪酬福利：
1. 具有竞争力的薪资
2. 年终奖金（13-16薪）
3. 股票期权

保险福利：
1. 五险一金
2. 补充商业保险
3. 年度体检

假期福利：
1. 法定节假日
2. 带薪年假（10-20天）
3. 病假、婚假、产假等

其他福利：
1. 下午茶
2. 健身房
3. 员工旅游
4. 节日礼品
5. 生日派对
""",
        "faq.txt": """常见问题

Q1: 如何联系技术支持？
A1: 发送邮件到 support@company.com，或拨打热线 400-123-4567。

Q2: API调用限制是多少？
A2: 免费版每月1000次，专业版10万次，企业版可定制。

Q3: 支持哪些支付方式？
A3: 支持支付宝、微信支付、对公转账。

Q4: 数据安全如何保障？
A4: 采用银行级加密，通过ISO27001认证，数据中心设在国内。

Q5: 是否支持私有化部署？
A5: 企业版支持私有化部署，具体方案请联系销售。
"""
    }
    
    for filename, content in docs.items():
        filepath = os.path.join(directory, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
    
    st.info(f"✅ 已创建 {len(docs)} 个示例文档到 {directory}")


def main():
    """主程序"""
    st.set_page_config(
        page_title="企业知识库问答系统",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("📚 企业知识库问答系统")
    st.markdown("基于 LangChain + Streamlit + SiliconFlow API (Qwen3-8B)")
    st.divider()
    
    # 初始化 session state
    if 'kb' not in st.session_state:
        st.session_state.kb = None
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # 侧边栏
    with st.sidebar:
        st.header("⚙️ 系统配置")
        
        # 初始化按钮
        if st.button("🚀 初始化知识库", use_container_width=True, type="primary"):
            with st.spinner("正在初始化知识库系统..."):
                st.session_state.kb = KnowledgeBase()
                
                # 检查是否有已有索引
                if os.path.exists(st.session_state.kb.index_directory) and os.listdir(st.session_state.kb.index_directory):
                    if st.session_state.kb.load_index():
                        st.success("✅ 成功加载已有知识库索引！")
                        st.session_state.initialized = True
                    else:
                        create_sample_docs()
                        if st.session_state.kb.build_index():
                            st.success("✅ 知识库构建成功！")
                            st.session_state.initialized = True
                        else:
                            st.error("❌ 知识库构建失败")
                else:
                    create_sample_docs()
                    if st.session_state.kb.build_index():
                        st.success("✅ 知识库构建成功！")
                        st.session_state.initialized = True
                    else:
                        st.error("❌ 知识库构建失败")
        
        st.divider()
        
        # 检索配置
        st.subheader("🔍 检索配置")
        k_value = st.slider("检索文档数量 (Top-K)", min_value=1, max_value=10, value=3)
        
        st.divider()
        
        # 系统状态
        st.subheader("📊 系统状态")
        if st.session_state.initialized:
            st.success("✅ 系统就绪")
            st.info(f"📄 文档目录: ./knowledge_docs")
            st.info(f"🗂️ 索引目录: ./knowledge_index")
        else:
            st.warning("⚠️ 未初始化，请点击上方按钮")
        
        st.divider()
        
        # 导出功能
        if st.session_state.initialized and st.session_state.kb.query_log:
            if st.button("📥 导出查询日志", use_container_width=True):
                st.session_state.kb.export_logs()
        
        st.divider()
        st.caption("💡 提示：使用 cpolar 可实现公网访问")
        st.caption("🎯 cpolar 命令: cpolar http 8501")
    
    # 主区域
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("💬 智能问答")
        
        # 显示聊天历史
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "sources" in message and message["sources"]:
                    with st.expander("📚 查看答案来源"):
                        for i, source in enumerate(message["sources"], 1):
                            st.markdown(f"**来源 {i}:** {source.metadata.get('source', '未知')}")
                            st.caption(f"内容片段: {source.page_content[:150]}...")
        
        # 输入框
        if st.session_state.initialized:
            question = st.chat_input("请输入您的问题...")
            
            if question:
                # 添加用户消息
                st.session_state.messages.append({"role": "user", "content": question})
                with st.chat_message("user"):
                    st.markdown(question)
                
                # 生成回答
                with st.chat_message("assistant"):
                    with st.spinner("🔍 正在检索并生成答案..."):
                        result = st.session_state.kb.query(question, k=k_value)
                        
                        if "error" not in result:
                            st.markdown(result["answer"])
                            
                            # 显示来源
                            if result.get("sources"):
                                with st.expander("📚 查看答案来源"):
                                    for i, doc in enumerate(result["sources"], 1):
                                        st.markdown(f"**来源 {i}:** {doc.metadata.get('source', '未知')}")
                                        st.caption(f"内容片段: {doc.page_content[:150]}...")
                            
                            # 保存消息
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": result["answer"],
                                "sources": result.get("sources", [])
                            })
                        else:
                            st.error(f"❌ {result['error']}")
        else:
            st.info("👈 请先在左侧点击【初始化知识库】按钮")
    
    with col2:
        st.subheader("📝 示例问题")
        example_questions = [
            "公司主要产品有哪些？",
            "AI开发平台的定价是多少？",
            "员工有哪些福利？",
            "如何联系技术支持？",
            "知识库问答系统有什么功能？"
        ]
        
        for q in example_questions:
            if st.button(q, use_container_width=True, key=q):
                if st.session_state.initialized:
                    # 自动填充问题
                    st.session_state.messages.append({"role": "user", "content": q})
                    with st.chat_message("user"):
                        st.markdown(q)
                    
                    with st.chat_message("assistant"):
                        with st.spinner("🔍 正在检索并生成答案..."):
                            result = st.session_state.kb.query(q, k=k_value)
                            if "error" not in result:
                                st.markdown(result["answer"])
                                st.session_state.messages.append({
                                    "role": "assistant",
                                    "content": result["answer"],
                                    "sources": result.get("sources", [])
                                })
                            else:
                                st.error(f"❌ {result['error']}")
                    st.rerun()
                else:
                    st.warning("请先初始化知识库")
        
        st.divider()
        
        # 统计信息
        if st.session_state.initialized and st.session_state.kb.query_log:
            st.subheader("📊 统计信息")
            st.metric("总提问次数", len(st.session_state.kb.query_log))


if __name__ == "__main__":
    main()