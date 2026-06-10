print("检查依赖包安装情况...")
print("-" * 40)

try:
    import streamlit
    print("✓ streamlit 已安装")
except ImportError:
    print("✗ streamlit 未安装")

try:
    import langchain
    print("✓ langchain 已安装")
except ImportError:
    print("✗ langchain 未安装")

try:
    import langchain_community
    print("✓ langchain_community 已安装")
except ImportError:
    print("✗ langchain_community 未安装")

try:
    import langchain_text_splitters
    print("✓ langchain_text_splitters 已安装")
except ImportError:
    print("✗ langchain_text_splitters 未安装")

try:
    import langchain_openai
    print("✓ langchain_openai 已安装")
except ImportError:
    print("✗ langchain_openai 未安装")

try:
    import faiss
    print("✓ faiss-cpu 已安装")
except ImportError:
    print("✗ faiss-cpu 未安装")

try:
    from dotenv import load_dotenv
    print("✓ python-dotenv 已安装")
except ImportError:
    print("✗ python-dotenv 未安装")

print("-" * 40)
print("检查完成！")