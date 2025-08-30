from app.core.config import settings
from langchain_openai import ChatOpenAI

from app.core.rag.prompt_templates import QueryExpansionTemplate


class QueryExpansion:
    # opik_tracer = OpikTracer(tags=["QueryExpansion"])

    @staticmethod
    # @opik.track(name="QueryExpansion.generate_response")
    def generate_response(query: str, to_expand_to_n: int, stream: bool | None = False) -> list[str]:
        query_expansion_template = QueryExpansionTemplate()
        prompt = query_expansion_template.create_template(to_expand_to_n)
        model = ChatOpenAI(
            model=settings.MODEL_PATH,
            api_key=settings.KEY,
            base_url=settings.LOCAL,
        )
        chain = prompt | model

        if stream:
            for chunk in chain.stream({"question": query}):
                # print(chunk, end="|", flush=True)
                yield chunk.content

        else:
            # chain = chain.with_config({"callbacks": [QueryExpansion.opik_tracer]})

            response = chain.invoke({"question": query})
            response = response.content

            queries = response.strip().split(query_expansion_template.separator)
            stripped_queries = [stripped_item for item in queries if (stripped_item := item.strip(" \\n"))]

            return stripped_queries
