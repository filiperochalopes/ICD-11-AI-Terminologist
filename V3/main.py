from V3.nodes.mapping import mapping_node
from V3.nodes.retrieval import retrieval_node
from V3.env import builder

# Add nodes
builder.add_node("retrieval", retrieval_node)
builder.add_node("mapping", mapping_node)

# Connect nodes and define entry/exit points
builder.set_entry_point("retrieval")
builder.add_edge("retrieval", "mapping")
builder.set_finish_point("mapping")

# Compile the graph
graph = builder.compile()