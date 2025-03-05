import pprint
from llama_index.core.response.notebook_utils import display_response

query_engine = index.as_query_engine(similarity_top_k=3)
response = query_engine.query(query)
display_response(response)
pprint.pprint(response.source_nodes)

"""### Reranking with VoyageAIRerank

Applies reranking to the search results using VoyageAIRerank, which aims to improve the relevance of the top results.
"""

from llama_index.postprocessor.voyageai_rerank import VoyageAIRerank

voyageai_rerank = VoyageAIRerank(
    api_key=voyage_api_key, top_k=2, model="rerank-2", truncation=True
)

"""### Query with Reranking

Queries the LlamaIndex index again, but this time with reranking applied using VoyageAIRerank. It displays the reranked response and the source nodes.
"""

query_engine = index.as_query_engine(
    similarity_top_k=10,
    node_postprocessors=[voyageai_rerank],
)
response = query_engine.query(query)
display_response(response)
pprint.pprint(response.source_nodes)