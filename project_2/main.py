from .research_assistant import AIResearchAssistant, print_research_response
from .data_model import ResearchResponse


if __name__ == "__main__":
    import shutil

    shutil.rmtree("./research_db", ignore_errors=True)
    assistant = AIResearchAssistant()

    # Add research docs
    assistant.add_text(
        """
        Attention Mechanisms in Neural Networks

        The attention mechanism was introduced in "Attention Is All You Need"
        by Vaswani et al. (2017). It allows models to focus on relevant parts
        of the input when generating output.

        Key concepts:
        - Query, Key, Value (QKV) triplets
        - Scaled dot-product attention
        - Multi-head attention for parallel processing

        The transformer architecture has become the foundation for modern NLP
        models including BERT, GPT, and T5.
        """,
        source="attention_mechanisms.pdf",
    )

    assistant.add_text(
        """
        Retrieval-Augmented Generation (RAG)

        RAG combines retrieval systems with generative models. First introduced
        by Lewis et al. (2020), RAG addresses the limitation of LLMs being
        limited to their training data.

        Components of a RAG system:
        1. Document store with vector embeddings
        2. Retriever to find relevant documents
        3. Generator (LLM) to produce responses

        Benefits include reduced hallucination, up-to-date information,
        and source attribution.
        """,
        source="rag_survey.pdf",
    )

    assistant.add_text(
        """
        LangChain and LangGraph Framework Overview

        LangChain is an open-source framework for building LLM applications.
        Key features include modular components, integration with 50+ LLM
        providers, and built-in RAG utilities.

        LangGraph extends LangChain for stateful applications with
        graph-based state management, support for cycles and loops,
        and human-in-the-loop workflows.
        """,
        source="langchain_docs.md",
    )

    print(f"\nIndexed: {assistant.get_document_count()} chunks")

    session = "structured_demo"

    # --- Step 1: String vs Structured comparison ---
    print("\n" + "=" * 60)
    print("STEP 1: String response vs Structured response")
    print("=" * 60)

    question = "What is RAG and what are its benefits?"

    print("\n--- String response (ask) ---")
    string_response = assistant.ask(question, "string_test")
    print(f"Type: {type(string_response)}")
    print(f"Response: {string_response[:200]}...")

    print("\n--- Structured response (ask_structured) ---")
    structured_response = assistant.ask_structured(question, "struct_test")
    print(f"Type: {type(structured_response)}")
    print(f"answer:             {structured_response.answer[:100]}...")
    print(f"confidence:         {structured_response.confidence}")
    print(f"sources:            {structured_response.sources}")
    print(f"key_quotes:         {structured_response.key_quotes[:2]}")
    print(f"follow_up_questions: {structured_response.follow_up_questions}")

    # --- Step 2: Access fields directly ---
    print("\n" + "=" * 60)
    print("STEP 2: Use fields in your code")
    print("=" * 60)

    r = assistant.ask_structured("What is the attention mechanism?", session)

    # This is what your app code looks like
    if r.confidence == "high":
        print(f"\n  Confident answer from: {', '.join(r.sources)}")
    else:
        print(f"\n  Low confidence -- may need more sources")

    print(f"\n  Answer: {r.answer[:200]}")

    print(f"\n  Suggested follow-ups:")
    for fq in r.follow_up_questions:
        print(f"    -> {fq}")

    # --- Step 3: Multi-turn with structured output ---
    print("\n" + "=" * 60)
    print("STEP 3: Memory works with structured output too")
    print("=" * 60)

    q1 = "What are the components of RAG?"
    print(f"\nUser: {q1}")
    r1 = assistant.ask_structured(q1, session)
    print_research_response(q1, r1)

    q2 = "How does the second component work?"
    print(f"\n{'- '*30}")
    print(f"\nUser: {q2}")
    r2 = assistant.ask_structured(q2, session)
    print_research_response(q2, r2)

    q3 = "Connect everything we discussed to LangChain."
    print(f"\n{'- '*30}")
    print(f"\nUser: {q3}")
    r3 = assistant.ask_structured(q3, session)
    print_research_response(q3, r3)

    # Cleanup
    shutil.rmtree("./research_db", ignore_errors=True)