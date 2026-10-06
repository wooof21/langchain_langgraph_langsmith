
## Course

https://www.udemy.com/course/production-ai-agents

## LangChian Documentation:
- Prompt Template: https://docs.langchain.com/langsmith/prompt-template-format#conversion-between-formats
- Prompt Engineering Concepts: https://docs.langchain.com/langsmith/prompt-engineering-concepts#prompts-vs-prompt-templates
- Runnables: https://reference.langchain.com/python/langchain-core/runnables
- Structured Output: 
    * agents: https://docs.langchain.com/oss/python/langchain/structured-output
    * model: https://docs.langchain.com/oss/python/langchain/models#structured-output
- RAG:
    * Retrieval: https://docs.langchain.com/oss/python/deepagents/retrieval
    * RAG: https://docs.langchain.com/oss/python/deepagents/rag
- Retriever: 
    * https://docs.langchain.com/oss/python/integrations/retrievers/index#retriever-integrations




## UV Package Manager

- https://docs.astral.sh/uv/#installation
- install(windows): powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

## LangSmith Monitoring & Tracing
- https://smith.langchain.com


## run:
Root folder: `uv run python -m 1_LangChain_Core_Concept_LCEL_Runnable_Chains.core_concepts`


## Content

1. LangChain Core Concept:
    * LCEL and Runnable Chains
    * Batch Execution
    * Stream Realtime Output with LCEL
    * Output Parser
2. Multi Provider Configuration:
    * Multi Providers Config
    * Multi Model Setup
3. Prompt Messages:
    * ChatPromptTemplate
    * FewShotChatMessagePromptTemplate
    * Reusable Components
4. Prompt Templates:
    * Message Types
    * Message Placeholder
    * FewShotChatMessagePromptTemplate
    * Prompt Composition
5. Output Parser:
    * StrOutputParser
    * JsonOutputParser
    * PydanticOutputParser
    * Structured Output
    * Complex Schema
6. Chains:
    * Basic Chain
    * Parallel Chain
    * Passthrough Chain
    * Chain Branching
7. Document Loading, Chunking & Embeddings
    * Text Loader
    * Web Loader
    * Lazy Loader
    * Document
    * PDF Loader
    * Text Splitter
        - RecursiveCharacterTextSplitter
        - MarkdownHeaderTextSplitter
        - Code Splitter
        - Document Splitter
    * Chunk Size
    * Chunk Overlap
8. RAG - Vector Embedding
    * OpenAI Embedding, HuggingFaceEmbeddings, OllamaEmbeddings
    * Basic Embedding: embed_query
    * Batch Embedding: embed_documents
    * Similarity Search
    * Embedding Caching
9. Vector Store - Chroma:
    * Similarity Search with Scores
    * Metadata Filtering
    * Chroma DB Persistence
    * Vector Store as Retriever for Chains
10. RAG Pipeline:
    * Basic RAG Pipeline
    * RAG with Sources
    * RAG with Fallback
    * RAG with Structured Output
11. Advanced RAG:
    * Multi Query Retriever
    * Contextual Compression - LLMChainExtractor(compressor) - ContextualCompressionRetriever
    * Hybrid Search: Keyword (BM25) & Semantic Search
        - BM25Retriever
        - EnsembleRetriever
    * Parent Document Retriever - ParentDocumentRetriever
12. Memory in LangChain:
    * Conversation Memory
    * Multiple Conversation Sessions Memory
    * Message Trimming
    * Sliding Window Memory
    * Summary Memory - Compress old message into Summary, maintain all necessary context
    * Persist Memory
13. LangGraph - Nodes
    * StateGraph
    * Accumulating State
    * Message State - The Chat Pattern
    * Multi-Node Graph - Chaining LLM Calls
    * Conversation Graph
14. LangGraph - Edges
    * Conditional Edges
    * Literal Types
    * Cycles and Loops in LangGraph
        - Self-Correcting Agent Pattern
        - Iterative Research
15. Human In The Loop
    * MemorySaver
    * Interrupt For Approval
    * Iterative Review
        - Human In The Loop + Cycles
    * Checkpointing and Persistence in LangGraph
        - Save and Resume Agent State
        - In-Memory Checkpointing
        - SqliteSaver - SQLite Persistence
        - PostgresSaver - Production
        - Branch Conversations from Checkpoints
16. Multi Agents System
    * Define Tool
        - ToolNode
        - ReAct Pattern
    * Custom Tool Error Handling
        - Return errors instead of raise them
    * Supervisor Pattern
    * Agent Handoffs
        - Triage Agent
    * Parallel Agent
        - Map Reduce Strategy
    * Agent Communication
        - Message Passing Pattern
        - Shared Field State
        - blackboard Pattern
17. Hierarchical Architecture
    * Hierarchical Routing
18. Production
    * Observability
        - Tracing: Named, Metadata
    * Security
        - Layer 1: Sanitizer - Input Sanitization
        - Layer 2: PII Detector
        - Layer 3: LLM-as-Guard
        - Layer 4: Own Agent 
        - Layey 5: Output Validation
    * Security Patterns Combined
        - Secure Pipeline
    * LLM Testing & Unit Testing
        - Mock Unit Testing
        - Integration Test
        - LLM as Judge: Use AI to Evaluate AI
        - Regression Testing
        - LangSmith Evaluation Datasets — Production Approach
        - Error Handling: 
            - Retry Decorator
            - Circuit Breaker
            - Model Fallback Chain
            - Graph Based Retry: LangGraph Error Handling
            - Reliability Patterns
        - Cost Optimization
            - Model Routing
            - Semantic Caching
                - Layer 1: Normalized Cache - convert to lowercase, strip whitespace
                - Layer 2: MD5 Hashing - hash normalized cache into fixed length key 
                    (cache lookup key)
            - Token Budgeting
    * Monitoring: Instrumented LLM
        - Structured Logging
        - Metrics Collection
        - Monitoring as Decorator: monitor_as_decorator

19. Project_1: Q&A Chat Bot
    * Chat Bot
        - Single Q&A
        - Batch Q&A
    * LangSimth Tracing
    * Error Handling

20. Project_2: Document Research Assistant
    * Document & Text Embeddings
    * Text Splitter
    * Vector Store
    * RAG Chain
    * Conversation Memory
    * Chat History
    * Structured Output

21. Project_3: Multi-Agent Research System
    * Hierarchical Architecture
    * Send API
    * Structured Output
    * Streaming Output



