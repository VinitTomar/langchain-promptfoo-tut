# LangChain.js & LangGraph.js — A Chapter-by-Chapter Walkthrough

A hands-on progression through [LangChain.js](https://js.langchain.com/docs/introduction/) and [LangGraph.js](https://langchain-ai.github.io/langgraphjs/), built one concept at a time. It starts with a single prompt sent to a model and ends with a persistent, multi-user chat application backed by Postgres.

Each chapter assumes the one before it. Read them in order and the ideas stack: a prompt becomes a chain, a chain becomes a graph, a graph gains memory, then branches, then tools, then other agents.

**Who this is for.** You're comfortable with LLMs in general — you know what a prompt is, roughly what an embedding is, and you've called a model API before. You have not used LangChain or LangGraph. Everything framework-specific is explained; everything else is assumed.

**Every example uses Google Gemini** (`gemini-2.0-flash` for chat, `gemini-embedding-001` for embeddings). No OpenAI or Anthropic keys are needed.

---

## Contents

- [How this repo works](#how-this-repo-works)
- [Setup](#setup)
- [Running the examples](#running-the-examples)
- [Chapter 1 — Prompting, LCEL, and evaluating with promptfoo](#chapter-1--prompting-lcel-and-evaluating-with-promptfoo)
- [Chapter 2 — The RAG ingest pipeline](#chapter-2--the-rag-ingest-pipeline)
- [Chapter 3 — Advanced RAG: query techniques](#chapter-3--advanced-rag-query-techniques)
- [Chapter 4 — Memory: state, checkpointers, and chat history](#chapter-4--memory-state-checkpointers-and-chat-history)
- [Chapter 5 — Cognitive architectures: call, chain, router](#chapter-5--cognitive-architectures-call-chain-router)
- [Chapter 6 — Agent architectures: tool calling](#chapter-6--agent-architectures-tool-calling)
- [Chapter 7 — Multi-agent patterns](#chapter-7--multi-agent-patterns)
- [Chapter 8 — Streaming and a console chat app](#chapter-8--streaming-and-a-console-chat-app)
- [Concept → file index](#concept--file-index)

---

## How this repo works

```
src/
├── shared/          base classes and helpers used by every chapter
├── utils.ts         prettyPrint() — colourised console output
├── chapter-1/       promptfoo-driven examples (see below)
└── chapter-2..8/    run.ts entry point + numbered example files
```

### `src/shared/` — read this first

Every example extends one of two base classes in [`src/shared/base-example-api-provider.ts`](src/shared/base-example-api-provider.ts):

- **`Example`** gives the subclass `this.model` — a `ChatGoogleGenerativeAI` instance already configured with your API key.
- **`ExampleWithEmbeddings`** adds `this.embeddings`, a `GeminiEmbeddings` instance.

Both require the subclass to implement `callApi(input: string)`. That signature comes from promptfoo's `ApiProvider` interface, which is why even the chapters that never touch promptfoo still use it — it's a consistent entry point.

The other shared helpers:

| File | Purpose |
|---|---|
| [`pg-db-conn.ts`](src/shared/pg-db-conn.ts) | Builds the Postgres connection string |
| [`pg-db-vector-store.ts`](src/shared/pg-db-vector-store.ts) | Wraps `PGVectorStore` and `PostgresRecordManager`. Note it suffixes every table name with a timestamp, so each run gets a fresh table |
| [`gemini-embeddings.ts`](src/shared/gemini-embeddings.ts) | `GoogleGenerativeAIEmbeddings` preconfigured with `gemini-embedding-001` |
| [`text-loader-with-recursive-splitter.ts`](src/shared/text-loader-with-recursive-splitter.ts) | Load a text file and split it in one call — used throughout chapter 3 |
| [`get-env-config.ts`](src/shared/get-env-config.ts) | Loads `.env` from the repo root |

### Chapter 1 is structured differently

Chapter 1 runs through [promptfoo](https://www.promptfoo.dev/docs/configuration/guide/), an LLM evaluation tool. Each example is a **four-file quartet**:

| File | Role |
|---|---|
| `<name>.ts` | The example itself, exported as a promptfoo custom provider |
| `<name>.prompt.txt` | The prompt — either literal text or a `{{variable}}` template |
| `<name>.config.yaml` | Glues prompt + provider + tests together |
| `<name>.test.yaml` | The assertions to check the output against |

### Chapters 2–8: pick an example by uncommenting it

Each of these chapters has a `run.ts` that imports every example and calls one of them. **You choose which example runs by editing the comments.** From [`src/chapter-6/run.ts`](src/chapter-6/run.ts):

```ts
(async () => {
  // await toolCallingAgent();
  // await toolCallFirstAgent();
  await multiToolHandlingAgent();
})();
```

**Several files have a second layer of the same pattern inside them.** After picking the file in `run.ts`, open it and check `callApi` — there is often another commented-out list choosing which technique runs:

| File | Choices inside |
|---|---|
| [`chapter-3/2-query-transformation.ts`](src/chapter-3/2-query-transformation.ts) | `retrieveRewriteRead` / `multiQueryRetrieval` / `ragFusion` / `hypotheticalDocumentEmbedding` |
| [`chapter-3/3-query-routing.ts`](src/chapter-3/3-query-routing.ts) | `logicalRouting` / `semanticRouting` |
| [`chapter-3/4-query-construction.ts`](src/chapter-3/4-query-construction.ts) | `textToMetadata` / `textToSQL` |
| [`chapter-4/2-modify-chat-history.ts`](src/chapter-4/2-modify-chat-history.ts) | `trimmingMsg` / `filterMessages` / `mergingConsecutiveMessages` |
| [`chapter-7/2-subgraph.ts`](src/chapter-7/2-subgraph.ts) | `directCall` / `indirectCall` |

Chapters 6 and 8 additionally have a commented-out `graph.invoke()` alternative next to the streaming path — swap them to see the final state instead of every intermediate step.

---

## Setup

### 1. Install dependencies

```bash
npm install --legacy-peer-deps
npm install --no-save --legacy-peer-deps better-sqlite3
```

Both lines are needed, and plain `npm install` will not work:

- **`--legacy-peer-deps`** — promptfoo requires `google-auth-library@^10`, while `@langchain/community@0.0.40` declares a peer range of `^8.9.0`. npm refuses to resolve that conflict on its own and aborts.
- **The second line** puts `better-sqlite3` at the top level, where promptfoo's copy of `drizzle-orm` can actually resolve it. Without it chapter 1 fails to start. `--no-save` leaves `package.json` unchanged.

**If the install fails while building `better-sqlite3`** with `ModuleNotFoundError: No module named 'distutils'`, the bundled `node-gyp` is too old for Python 3.12+, which removed `distutils` from the standard library. Point npm at any Python 3.11 or older that you have installed, and use it for both lines:

```bash
npm_config_python=python3.10 npm install --legacy-peer-deps
npm_config_python=python3.10 npm install --no-save --legacy-peer-deps better-sqlite3
```

### 2. Get a Gemini API key

Create one at [Google AI Studio](https://aistudio.google.com/app/apikey), then copy the template and fill it in:

```bash
cp .env.example .env
```

```
GOOGLE_API_KEY=your-key-here
```

The `.env` file must sit at the **repo root**, next to `package.json` — [`src/shared/get-env-config.ts`](src/shared/get-env-config.ts) resolves it relative to the source file, so it is found no matter which directory you run from.

That's everything you need for **chapters 1, 4, 5 and 7**. The remaining prerequisites are per-chapter.

### 3. Postgres with pgvector — needed for chapters 2, 3 and 8

[`src/shared/pg-db-conn.ts`](src/shared/pg-db-conn.ts) hardcodes the connection string, so the container has to match it exactly:

```bash
docker run -d --name pg-vector-store \
  -e POSTGRES_USER=dev_user \
  -e POSTGRES_PASSWORD=dev_password \
  -e POSTGRES_DB=langchain_db \
  -p 5432:5432 \
  pgvector/pgvector:pg17
```

Keep the `pg17` tag. The [pgvector](https://github.com/pgvector/pgvector) image publishes no `latest` tag, so an unpinned `pgvector/pgvector` will fail to pull.

**Then add a hosts entry.** This step is required, not optional. The code connects to the hostname `pg-vector-store`, not `localhost`, so that name has to resolve on the machine running the examples. Without it you get an opaque DNS error on every Postgres-backed chapter:

```bash
echo "127.0.0.1 pg-vector-store" | sudo tee -a /etc/hosts
```

(`/etc/hosts` on macOS and Linux; `C:\Windows\System32\drivers\etc\hosts` on Windows.)

Verify it works:

```bash
psql postgresql://dev_user:dev_password@pg-vector-store:5432/langchain_db -c 'select 1'
```

### 4. A Tavily key — needed for parts of chapter 6

Examples 2 and 3 in chapter 6 search the web through [Tavily](https://www.tavily.com/). Add the key to `.env`; LangChain picks it up from the environment automatically:

```
TAVILY_API_KEY=your-key-here
```

Chapter 6's first example uses DuckDuckGo instead and needs no key.

### 5. `chinook.db` — needed for one example in chapter 3

The `textToSQL` half of [`chapter-3/4-query-construction.ts`](src/chapter-3/4-query-construction.ts) queries the Chinook sample database. Download `Chinook_Sqlite.sqlite` from the [Chinook repository](https://github.com/lerocha/chinook-database/releases), rename it to `chinook.db`, and place it in `src/chapter-3/files/`.

---

## Running the examples

### Chapter 1

```bash
npm run test:ch1 <example-name>
```

promptfoo compiles the TypeScript provider itself, so no separate runner is involved. The seven valid names — note the capitalisation of `LCEL-example`:

```
simple-example
usable-prompt-template
usable-chat-prompt-template
chat-model
LCEL-example
runnable-interface
specific-output-format
```

For example:

```bash
npm run test:ch1 simple-example
```

### Chapters 2–8

```bash
cd src/chapter-4 && npx tsx run.ts
```

**The `cd` is required, not stylistic.** These files load data with relative paths like `./files/test.txt`, which resolve against the working directory rather than the source file's location. Running from the repo root will not find them.

Chapter 4 is the easiest starting point — it needs only `GOOGLE_API_KEY`.

---

## Chapter 1 — Prompting, LCEL, and evaluating with promptfoo

**The idea.** Everything in this repo is built from one primitive: hand a model some messages, get one back. This chapter covers the ways LangChain lets you construct those messages — a raw string, a `PromptTemplate` with slots, a `ChatPromptTemplate` of system/human turns, or hand-built message objects — and then the thing that makes LangChain more than a wrapper: `.pipe()`. Piping a template into a model produces a *chain*, and every chain implements the same `Runnable` interface, so it can be invoked, streamed, or batched exactly like the model itself.

The chapter also introduces evaluation. Rather than eyeballing output, each example is wrapped as a promptfoo custom provider and checked against assertions — `contains` for substrings, `is-json` for structure. That is what the `.test.yaml` files hold.

**Examples** (one row per four-file quartet):

| Example | Demonstrates |
|---|---|
| `simple-example` | The minimum viable provider — `model.invoke(prompt)`, return `{ output }` |
| `usable-prompt-template` | `PromptTemplate.fromTemplate` with `{context}` and `{question}` slots, plus an instruction to answer "I don't know" when the context doesn't cover it |
| `usable-chat-prompt-template` | The same task expressed as a system/human message list instead of one flat string |
| `chat-model` | Building `SystemMessage` and `HumanMessage` objects by hand; the system prompt steers formatting ("four exclamation marks") |
| `LCEL-example` | `template.pipe(this.model)` — composition, the heart of LCEL |
| `runnable-interface` | `RunnableLambda.from(async function* …)` wrapping `model.stream()` to yield tokens as they arrive |
| `specific-output-format` | `model.withStructuredOutput(zodSchema)` returns a typed object instead of prose |

**New APIs:** `PromptTemplate`, `ChatPromptTemplate`, `SystemMessage`, `HumanMessage`, `.pipe()`, `RunnableLambda`, `.stream()`, `.withStructuredOutput()`, Zod schemas.

**Docs:** [Prompt templates](https://js.langchain.com/docs/concepts/prompt_templates) · [Chat models](https://js.langchain.com/docs/concepts/chat_models) · [LCEL](https://js.langchain.com/docs/concepts/lcel) · [Runnables](https://js.langchain.com/docs/concepts/runnables) · [Structured output](https://js.langchain.com/docs/concepts/structured_outputs) · [promptfoo custom providers](https://www.promptfoo.dev/docs/providers/custom-api/) · [promptfoo assertions](https://www.promptfoo.dev/docs/configuration/expected-outputs/)

---

## Chapter 2 — The RAG ingest pipeline

**The idea.** A model knows two things: what's in its weights, and what you put in the prompt. Retrieval-Augmented Generation is a way to control the second — fetch relevant text at question time and paste it into the prompt. But before you can fetch anything, documents have to get *in*. That's this chapter, in pipeline order: load from a source, split into chunks small enough to be useful as context, convert each chunk to a vector, store the vectors so they can be searched, and finally keep the store in sync as documents change.

Splitting matters more than it looks. Chunks that are too large waste context and dilute the match; too small and they lose the meaning that made them relevant. `RecursiveCharacterTextSplitter` splits on progressively finer separators — paragraphs, then lines, then words — so it breaks at natural boundaries where it can.

**Files**

| File | Demonstrates |
|---|---|
| [`1-text-loader.ts`](src/chapter-2/1-text-loader.ts) | `TextLoader` turning `files/test.txt` into an array of `Document` objects |
| [`2-web-base-loader.ts`](src/chapter-2/2-web-base-loader.ts) | `CheerioWebBaseLoader` scraping a live page |
| [`3-pdf-loader.ts`](src/chapter-2/3-pdf-loader.ts) | `PDFLoader` over `files/test.pdf` |
| [`4-recursive-text-splitter.ts`](src/chapter-2/4-recursive-text-splitter.ts) | `RecursiveCharacterTextSplitter` (chunk size 100, overlap 20), and the syntax-aware `fromLanguage('python', …)` variant that splits code on function boundaries |
| [`5-text-embedding.ts`](src/chapter-2/5-text-embedding.ts) | `embedDocuments()` — chunks in, float vectors out |
| [`6-store-embedding-pg-vector.ts`](src/chapter-2/6-store-embedding-pg-vector.ts) | Writing vectors into pgvector, then `similaritySearch('LangSmith', 3)` to read the nearest three back |
| [`7-manage-docs-lifecycle.ts`](src/chapter-2/7-manage-docs-lifecycle.ts) | `index()` with a `PostgresRecordManager` and `cleanup: 'incremental'` |

The last one is the most interesting. It indexes the same two documents three times: the first pass adds them, the second skips both because nothing changed, and the third — after mutating one document's content — updates just that one. Watch `numAdded` / `numSkipped` / `numUpdated` change across the three printouts. That's how you re-index a corpus without re-embedding everything.

**Requires:** Postgres (examples 6 and 7).

**New APIs:** `TextLoader`, `CheerioWebBaseLoader`, `PDFLoader`, `RecursiveCharacterTextSplitter`, `embedDocuments()`, `PGVectorStore`, `similaritySearch()`, `index()`, `PostgresRecordManager`.

**Docs:** [Document loaders](https://js.langchain.com/docs/concepts/document_loaders) · [Text splitters](https://js.langchain.com/docs/concepts/text_splitters) · [Embedding models](https://js.langchain.com/docs/concepts/embedding_models) · [Vector stores](https://js.langchain.com/docs/concepts/vectorstores) · [pgvector integration](https://js.langchain.com/docs/integrations/vectorstores/pgvector) · [Indexing](https://js.langchain.com/docs/how_to/indexing)

---

## Chapter 3 — Advanced RAG: query techniques

**The idea.** Chapter 2 got documents into the store. This chapter gets the *right ones* back out, and it starts from an uncomfortable premise: the user's question is usually a bad search query. It's vague, or it's phrased differently from the documents, or the real question is buried under three sentences of irrelevant preamble. Every technique here either transforms the query before it reaches the vector store, or decides which store to send it to at all.

The test fixture is worth knowing about. [`files/sample_greek_philosophy.txt`](src/chapter-3/files/sample_greek_philosophy.txt) deliberately contains **invented** Pre-Socratic philosopher names. If the answer comes back with those invented names, retrieval genuinely worked. If it names real philosophers, the model is answering from memory and ignoring your context — a failure you'd never notice with a factually correct corpus.

**Files**

| File | Demonstrates |
|---|---|
| [`1-rag-intro.ts`](src/chapter-3/1-rag-intro.ts) | The baseline: retrieve chunks, join them into `{context}`, answer from that alone |
| [`2-query-transformation.ts`](src/chapter-3/2-query-transformation.ts) | Four ways to rewrite the query — see below |
| [`3-query-routing.ts`](src/chapter-3/3-query-routing.ts) | **Logical routing** (`withStructuredOutput` + a Zod enum picks `python_docs` or `js_docs`) versus **semantic routing** (embed several candidate system prompts, `cosineSimilarity` against the query, use the closest) |
| [`4-query-construction.ts`](src/chapter-3/4-query-construction.ts) | Turning text into a *structured* query — `SelfQueryRetriever` + `AttributeInfo` infers metadata filters from "movies rated higher than 8.5"; `createSqlQueryChain` + `QuerySqlTool` generates and runs SQL |

The four strategies inside `2-query-transformation.ts`:

| Method | What it does |
|---|---|
| `retrieveRewriteRead` | Ask the model for a cleaner search query first, then retrieve with that |
| `multiQueryRetrieval` | Generate five paraphrases, `retriever.batch()` all of them, dedupe the union by page content |
| `ragFusion` | Generate four paraphrases, then merge their ranked result lists with reciprocal rank fusion — each document scores `1 / (rank + 60)` summed across lists, so a document ranked moderately by every paraphrase beats one ranked first by a single paraphrase |
| `hypotheticalDocumentEmbedding` | HyDE: have the model *invent* an answer, embed that passage, and search with it — on the theory that a hypothetical answer sits closer in embedding space to the real answer than the question does |

Its test question buries the real question under noise about brushing teeth and forgetting food on the cooker. That's the point — it's what makes rewriting worth the extra model call.

**Requires:** Postgres. The `textToSQL` example additionally needs `chinook.db`.

**New APIs:** `cosineSimilarity`, `SelfQueryRetriever`, `AttributeInfo`, `FunctionalTranslator`, `MemoryVectorStore`, `SqlDatabase`, `createSqlQueryChain`, `QuerySqlTool`, `retriever.batch()`.

**Docs:** [Retrieval](https://js.langchain.com/docs/concepts/retrieval) · [Retrievers](https://js.langchain.com/docs/concepts/retrievers) · [Multiple query retriever](https://js.langchain.com/docs/how_to/multiple_queries) · [HyDE](https://js.langchain.com/docs/how_to/hyde) · [Self-query retriever](https://js.langchain.com/docs/how_to/self_query) · [SQL question answering](https://js.langchain.com/docs/tutorials/sql_qa)

---

## Chapter 4 — Memory: state, checkpointers, and chat history

**The idea.** This is the first LangGraph chapter, and the shift in mental model matters. A chain is one-shot: input goes in, output comes out, nothing persists. A conversation needs state that survives between turns.

LangGraph models an application as a graph of nodes over a **shared state object**. You declare the shape of that state with `Annotation.Root`, each node receives the current state and returns a partial update, and a **reducer** decides how updates merge in. `messagesStateReducer` is the one you'll use constantly — it *appends* to the message list rather than replacing it, which is exactly what a transcript needs.

A **checkpointer** then saves that state after every step, keyed by `thread_id`. Pass the same `thread_id` on the next call and the graph resumes where it left off. That single idea is what turns a stateless graph into a chatbot.

The second half is the unglamorous consequence: history grows without bound, and eventually exceeds the context window. So you trim it.

**Files**

| File | Demonstrates |
|---|---|
| [`1-chat-memory-system.ts`](src/chapter-4/1-chat-memory-system.ts) | The same one-node graph compiled twice — `simpleGraph` without a checkpointer, `simpleGraphWithMemory` with `MemorySaver`. The memory version says "Hi, my name is Vinit", then asks "what is my name?" on the same `thread_id`. Only one of them can answer |
| [`2-modify-chat-history.ts`](src/chapter-4/2-modify-chat-history.ts) | `trimMessages` (cap at 30 tokens, `strategy: 'last'`, `startOn: 'human'` so the window never opens mid-exchange), `filterMessages` (keep by type, drop by ID), and `mergeMessageRuns` (collapse consecutive same-role messages into one) |

**New APIs:** `StateGraph`, `Annotation.Root`, `messagesStateReducer`, `START` / `END`, `.compile({ checkpointer })`, `MemorySaver`, `thread_id`, `trimMessages`, `filterMessages`, `mergeMessageRuns`.

**Docs:** [LangGraph low-level concepts](https://langchain-ai.github.io/langgraphjs/concepts/low_level/) · [Persistence](https://langchain-ai.github.io/langgraphjs/concepts/persistence/) · [Trim messages](https://js.langchain.com/docs/how_to/trim_messages) · [Filter messages](https://js.langchain.com/docs/how_to/filter_messages) · [Merge message runs](https://js.langchain.com/docs/how_to/merge_message_runs)

---

## Chapter 5 — Cognitive architectures: call, chain, router

**The idea.** Once you can build a graph, the design question becomes: what shape should it be? This chapter walks three shapes in order of how much control you hand over to the model — from a path you fully determine to one the model chooses at runtime.

It also introduces **custom state channels**. Chapter 4's state held only `messages`; here the state gains fields like `userQuery`, `sqlQuery` and `sqlExplain`. Graph state is your application's working memory, and it can hold whatever your nodes need to pass to each other.

**Files**

| File | Demonstrates |
|---|---|
| [`1-llm-call.ts`](src/chapter-5/1-llm-call.ts) | A single node. The graph adds nothing but structure — the baseline the other two are measured against |
| [`2-llm-chain.ts`](src/chapter-5/2-llm-chain.ts) | Two fixed nodes wired in sequence, `generate_sql` → `explain_sql`, each with its own system prompt and its own temperature (0.1 for generating, 0.7 for explaining). You decide the path, not the model |
| [`3-llm-router.ts`](src/chapter-5/3-llm-router.ts) | The model classifies the query into `records` or `insurance`; `addConditionalEdges` dispatches to the matching retrieval branch, and both branches converge on one shared `generate_answer` node |

The router is the important one. `addConditionalEdges` takes a function that inspects state and returns the name of the next node — that's the mechanism behind every branching pattern in the remaining chapters.

**New APIs:** `addConditionalEdges`, custom `Annotation` state channels, per-node model configuration, `graph.stream()`.

**Docs:** [LangGraph low-level concepts](https://langchain-ai.github.io/langgraphjs/concepts/low_level/) · [Workflows and agents](https://langchain-ai.github.io/langgraphjs/concepts/agentic_concepts/)

---

## Chapter 6 — Agent architectures: tool calling

**The idea.** A router picks between branches you defined in advance. An **agent** decides how many steps to take, and which, at runtime. The loop is short: call the model, check whether it asked for a tool, run the tool, feed the result back, repeat until it answers instead of asking.

LangGraph ships both halves prebuilt. `ToolNode` is a node that executes whatever tool calls it finds on the last message. `toolsCondition` is a conditional edge that inspects the last message and routes either to the tools node or to the end. Wire `model → toolsCondition` and `tools → model` and you have the classic ReAct cycle in four lines.

**Files**

| File | Demonstrates |
|---|---|
| [`1-tool-calling-agent.ts`](src/chapter-6/1-tool-calling-agent.ts) | The standard loop — `model ⇄ tools`, with DuckDuckGo search wrapped via `tool()` and a `Calculator` |
| [`2-tool-call-first-agent.ts`](src/chapter-6/2-tool-call-first-agent.ts) | Forcing a tool call *before* the model's first turn. A node fabricates an `AIMessage` carrying a synthetic `tool_calls` entry, so search runs and its results are in context before the model ever sees the question |
| [`3-multi-tool-handling-agent.ts`](src/chapter-6/3-multi-tool-handling-agent.ts) | Tool **selection**. Each tool's `description` is embedded into a `MemoryVectorStore`; a `select_tool` node retrieves the ones relevant to this query and only those get bound. This is the answer to "what happens when I have 200 tools and they don't fit in the prompt?" |

**Requires:** `TAVILY_API_KEY` for examples 2 and 3. Example 1 uses DuckDuckGo and needs no key.

**New APIs:** `ToolNode`, `toolsCondition`, `tool()`, `bindTools()`, `DuckDuckGoSearch`, `TavilySearchResults`, `Calculator`, synthetic `tool_calls` on an `AIMessage`.

**Docs:** [Agentic concepts](https://langchain-ai.github.io/langgraphjs/concepts/agentic_concepts/) · [Tool calling in LangGraph](https://langchain-ai.github.io/langgraphjs/how-tos/tool-calling/)

---

## Chapter 7 — Multi-agent patterns

**The idea.** One agent with a long system prompt and twenty tools eventually stops being the right shape — the instructions conflict, and the model loses track of what it's doing. The fix is to split the work across several model calls with distinct roles, and let the graph structure decide how they hand off to each other. Three patterns, each solving a different coordination problem.

**Files**

| File | Demonstrates |
|---|---|
| [`1-reflection.ts`](src/chapter-7/1-reflection.ts) | **Generator/critic.** A `generate` node writes an essay, a `review` node grades it, and the edge loops back until `messages.length > 6`. Note the `clsMap` trick: before the critic runs, the essay is relabelled from `AIMessage` to `HumanMessage`, so the critic reads it as a submission to grade rather than as its own prior output |
| [`2-subgraph.ts`](src/chapter-7/2-subgraph.ts) | **Nesting graphs**, two ways. `directCall` passes the compiled subgraph straight in as a node — which works only when parent and child share a state key. `indirectCall` wraps it in a function that translates between two different state shapes, which is what you need in practice |
| [`3-supervisor.ts`](src/chapter-7/3-supervisor.ts) | **Supervisor.** A coordinator node uses `withStructuredOutput` over a Zod enum (`FINISH`, `researcher`, `coder`) to name the next worker. Workers do their job and hand control back, and the supervisor decides again — a loop that terminates when it returns `FINISH` |

**New APIs:** compiled graphs as nodes, state translation wrappers, `MessagesAnnotation.spec`, supervisor routing via `withStructuredOutput` + Zod enum.

**Docs:** [Multi-agent systems](https://langchain-ai.github.io/langgraphjs/concepts/multi_agent/) · [Reflection tutorial](https://langchain-ai.github.io/langgraphjs/tutorials/reflection/reflection/) · [Agent supervisor tutorial](https://langchain-ai.github.io/langgraphjs/tutorials/multi_agent/agent_supervisor/) · [Using subgraphs](https://langchain-ai.github.io/langgraphjs/how-tos/subgraph/)

---

## Chapter 8 — Streaming and a console chat app

**The idea.** Two halves. First, observability: a graph that runs to completion and prints one answer is impossible to debug, so LangGraph exposes several ways to watch it work. `graph.stream()` with different `streamMode` values yields either full state snapshots or per-node updates; `graph.streamEvents()` gives a much finer feed, down to individual tokens and callback events.

Then the capstone — everything from the previous seven chapters assembled into an application someone could actually use.

**Files**

| File | Demonstrates |
|---|---|
| [`intermediate-output.ts`](src/chapter-8/intermediate-output.ts) | `graph.stream(input, { streamMode })` across the `StreamMode` values, versus `graph.streamEvents(input, { version: 'v2' })` |
| [`console-ai-chat.ts`](src/chapter-8/console-ai-chat.ts) | The full application — signup and login with masked password entry, multiple named chat threads per user keyed by UUID, `chalk`-coloured transcripts, and `PostgresSaver` as the checkpointer so conversations survive a process restart. Reopening a thread replays its history via `graph.getState()` |
| [`pg-db-repo.ts`](src/chapter-8/pg-db-repo.ts) | Plain `pg` `Pool` CRUD for the `users` and `user_chat_threads` tables; both self-create with `CREATE TABLE IF NOT EXISTS` |

`PostgresSaver` is the payoff for chapter 4. Swapping `MemorySaver` for it is a one-line change, and it's the difference between a chatbot that forgets on restart and one that doesn't.

There's a comment worth reading at [`console-ai-chat.ts:85`](src/chapter-8/console-ai-chat.ts#L85) on the observed behavioural difference between `new SystemMessage(...)` and the `['system', '...']` tuple form — the tuple form turns out to follow instructions more strictly.

**Requires:** Postgres.

**New APIs:** `StreamMode`, `graph.streamEvents()`, `graph.getState()`, `PostgresSaver.fromConnString()`.

**Docs:** [Streaming](https://langchain-ai.github.io/langgraphjs/concepts/streaming/) · [Stream values](https://langchain-ai.github.io/langgraphjs/how-tos/stream-values/) · [Postgres persistence](https://langchain-ai.github.io/langgraphjs/how-tos/persistence-postgres/)

---

## Concept → file index

| Concept | Where |
|---|---|
| Calling a chat model | `chapter-1/simple-example.ts` |
| Prompt templates with variables | `chapter-1/usable-prompt-template.ts` |
| System / human message lists | `chapter-1/usable-chat-prompt-template.ts`, `chapter-1/chat-model.ts` |
| LCEL composition with `.pipe()` | `chapter-1/LCEL-example.ts` |
| The Runnable interface | `chapter-1/runnable-interface.ts` |
| Token streaming from a model | `chapter-1/runnable-interface.ts` |
| Structured output with Zod | `chapter-1/specific-output-format.ts`, `chapter-3/3-query-routing.ts` |
| Evaluating output with assertions | `chapter-1/*.test.yaml` |
| Loading text, web pages and PDFs | `chapter-2/1-text-loader.ts`, `chapter-2/2-web-base-loader.ts`, `chapter-2/3-pdf-loader.ts` |
| Chunking documents | `chapter-2/4-recursive-text-splitter.ts` |
| Splitting source code by syntax | `chapter-2/4-recursive-text-splitter.ts` |
| Generating embeddings | `chapter-2/5-text-embedding.ts` |
| Storing vectors in pgvector | `chapter-2/6-store-embedding-pg-vector.ts` |
| Similarity search | `chapter-2/6-store-embedding-pg-vector.ts` |
| Incremental re-indexing | `chapter-2/7-manage-docs-lifecycle.ts` |
| Baseline RAG chain | `chapter-3/1-rag-intro.ts` |
| Rewrite-retrieve-read | `chapter-3/2-query-transformation.ts` |
| Multi-query retrieval | `chapter-3/2-query-transformation.ts` |
| RAG-Fusion / reciprocal rank fusion | `chapter-3/2-query-transformation.ts` |
| HyDE | `chapter-3/2-query-transformation.ts` |
| Logical routing | `chapter-3/3-query-routing.ts` |
| Semantic routing with cosine similarity | `chapter-3/3-query-routing.ts` |
| Text → metadata filter | `chapter-3/4-query-construction.ts` |
| Text → SQL | `chapter-3/4-query-construction.ts` |
| Building a `StateGraph` | `chapter-4/1-chat-memory-system.ts` |
| State annotations and reducers | `chapter-4/1-chat-memory-system.ts` |
| Checkpointers and `thread_id` | `chapter-4/1-chat-memory-system.ts` |
| Trimming, filtering, merging history | `chapter-4/2-modify-chat-history.ts` |
| Sequential multi-node chain | `chapter-5/2-llm-chain.ts` |
| Custom state channels | `chapter-5/2-llm-chain.ts` |
| Conditional edges / branching | `chapter-5/3-llm-router.ts` |
| ReAct tool-calling loop | `chapter-6/1-tool-calling-agent.ts` |
| `ToolNode` and `toolsCondition` | `chapter-6/1-tool-calling-agent.ts` |
| Forcing a tool call first | `chapter-6/2-tool-call-first-agent.ts` |
| Selecting tools by embedding | `chapter-6/3-multi-tool-handling-agent.ts` |
| Reflection / generator-critic loop | `chapter-7/1-reflection.ts` |
| Subgraphs | `chapter-7/2-subgraph.ts` |
| Supervisor pattern | `chapter-7/3-supervisor.ts` |
| Stream modes and `streamEvents` | `chapter-8/intermediate-output.ts` |
| Persistent chat with `PostgresSaver` | `chapter-8/console-ai-chat.ts` |
| Reading graph state | `chapter-8/console-ai-chat.ts` |
