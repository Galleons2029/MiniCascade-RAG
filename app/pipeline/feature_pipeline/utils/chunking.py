from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)

from app.pipeline.feature_pipeline.config import settings


def chunk_text(text: str) -> list[str]:
    """
    使用递归分词，详见：
    https://python.langchain.com/docs/how_to/recursive_text_splitter/
    """
    character_splitter = RecursiveCharacterTextSplitter(separators=["\n\n"], chunk_size=500, chunk_overlap=0)
    text_split = character_splitter.split_text(text)

    token_splitter = SentenceTransformersTokenTextSplitter(
        chunk_overlap=50,
        tokens_per_chunk=settings.EMBEDDING_MODEL_MAX_INPUT_LENGTH,
        model_name="/data/model_cache/bge-m3",
    )
    chunks = []

    for section in text_split:
        chunks.extend(token_splitter.split_text(section))

    return chunks


def main():
    st = """
    为大型语言模型（LLMs）添加上下文能显著提升各类应用场景的性能表现。尽管检索增强生成（RAG）系统已得到广泛研究，但核心问题仍未解决：错误究竟源于LLM未能有效利用检索到的上下文，还是上下文本身不足以回答查询？
为探究这一问题，我们提出了充分上下文的新概念，并开发了相应的分类方法以判定实例是否包含足够信息来回答问题。基于这一标准，我们对多个模型和数据集展开分析。通过按上下文充分性对错误进行分层，研究发现：基线性能更高的大模型（如Gemini 1.5 Pro、GPT-4o、Claude 3.5）在上下文充分时表现优异，但当上下文不足时却倾向于输出错误答案而非主动弃答；而基线性能较弱的小模型（如Mistral 3、Gemma 2）即使面对充分上下文也经常产生幻觉或过度弃答。我们进一步识别出"部分有效上下文"场景——虽然上下文不能完全解答问题，但能提升模型准确率（若无上下文模型必定出错）。
基于这些发现，我们探索了减少RAG系统幻觉的方法，包括一种新型的选择性生成技术：利用充分上下文信息引导模型主动弃答。实验表明，该方法能将Gemini、GPT和Gemma系列模型在响应时的正确答案比例提升2%-10%。  
    """  # noqa: E501

    CHUNKS = chunk_text(st)
    print(CHUNKS)


if __name__ == "__main__":
    main()
