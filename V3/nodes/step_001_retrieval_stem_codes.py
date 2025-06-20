from V3.classes import GraphState, GraphStateManager
from V3.tools import vector_database_retrieve_stem_codes


def step_001_retrieval_stem_codes(state: GraphState) -> GraphState:
    """
    Node Step: Semantic Retrieval of Stem Codes from Qdrant

    - Takes the user's last message as the clinical concept input.
    - Runs semantic search using the qdrant_tool to retrieve relevant stem codes.
    - Appends the Qdrant search results as an AI message for traceability.
    - Initializes an empty memory list for downstream tools.
    - Returns the updated state dictionary.
    """

    # Extract the last user message from the conversation history
    user_message = state.messages[-1].content
    print(f"🔍 User message for retrieval: {user_message}")

    sm = GraphStateManager(state)

    # Run semantic search against Qdrant using the user input as query
    result = vector_database_retrieve_stem_codes._run(
        sm.update({"clinical_concept_input": user_message})
    )

    # Return updated graph state
    return GraphStateManager(result).update(
        {
            "task_memory": [
                {
                    "name": "step",
                    "content": "step_001_retrieval_stem_codes",
                }
            ]
        }
    )
