from dotenv import load_dotenv
load_dotenv()

from codegen_agent.config import get_llm, get_checkpointer
from codegen_agent.graph import build_graph

llm          = get_llm()
checkpointer = get_checkpointer()
graph        = build_graph(llm, checkpointer, use_doc_retriever=False)

# Save as PNG using LangGraph's built-in Mermaid renderer
png_bytes = graph.get_graph().draw_mermaid_png()
with open("graph.png", "wb") as f:
    f.write(png_bytes)

print("✅ graph.png saved")