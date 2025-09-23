import { ProviderResponse } from 'promptfoo';
import { RunnableLambda } from '@langchain/core/runnables';
import { VectorStoreRetriever } from '@langchain/core/vectorstores';
import { PGVectorStore } from '@langchain/community/vectorstores/pgvector';
import { Example } from '../shared/base-example-api-provider';
import { PgDbVectorStore } from '../shared/pg-db-vector-store';
import { TextLoaderWithRecursiveSplitter } from '../shared/text-loader-with-recursive-splitter';
import { ChatPromptTemplate } from '@langchain/core/prompts';
import { Documents, prettyPrint } from '../utils';
import { AIMessageChunk } from '@langchain/core/messages';
import { Document } from '@langchain/core/documents';

type PromptInputs = { context: string, question: string };


export class QueryTransformation extends Example {
  private retriever!: VectorStoreRetriever<PGVectorStore>;
  
  private async initRetriever() {
    const loader = new TextLoaderWithRecursiveSplitter(
      './files/sample_greek_philosophy.txt',
      {
        chunkSize: 500,
        chunkOverlap: 200,
      }
    );

    const docs = await loader.loadSplittedDocs();
    this.retriever = await PgDbVectorStore.getVectorStoreAsRetriever({
      docs,
      tableName: `query_transformation`
    });
  }

  retrieveRewriteRead(prompt: ChatPromptTemplate<PromptInputs>) {
    const rewritePrompt = ChatPromptTemplate.fromTemplate(
      `Provide a better search query for web search engine to answer the given question, end the queries with ’**’.

      Question: {question}

      Answer:
      `
    );

    const rewriter = rewritePrompt.pipe(this.model).pipe(
      (msg: AIMessageChunk) => {
        return msg.content.toString().replaceAll('"', '').replaceAll('**', '');
      }
    );
    
    return async (input: string) => {
       const newQuery = await rewriter.invoke({ question: input });
        const docs = await this.retriever.invoke(newQuery);
        const llmPrompt = await prompt.invoke({
          context: docs.map(doc => doc.pageContent).join(' '),
          question: input
        });
  
        return await this.model.invoke(llmPrompt);
    }
  }

  multiQueryRetrieval(prompt: ChatPromptTemplate<PromptInputs>) {
    const perspectivesPrompt = ChatPromptTemplate.fromTemplate(`You are an AI 
      language model assistant. Your task is to generate five different versions 
      of the given user question to retrieve relevant documents from a vector 
      database. By generating multiple perspectives on the user question, your 
      goal is to help the user overcome some of the limitations of the 
      distance-based similarity search. Provide these alternative questions 
      separated by newlines. Original question: {question}`);

    const queryGen = perspectivesPrompt
      .pipe(this.model)
      .pipe((message: AIMessageChunk) => {
        return message.content.toString()
          .split('\n')
          .map(s => s.replaceAll('\\', '')).filter(s => !!s);
      });
    
    const retrievalChain = queryGen
      .pipe(async (inp) => {
        return await this.retriever.batch(
          inp
        );
      })
      .pipe((documentLists) => {
        const dedupedDocs: Record<string, Document> = {};
        documentLists.flat().forEach(doc => {
          dedupedDocs[doc.pageContent] = doc;
        })
        return Object.values(dedupedDocs)
      });
    
    return async (input: string) => {
      const docs = await retrievalChain.invoke({ question: input });
      const finalPrompt = await prompt.invoke({
        context: docs.map(d => d.pageContent.toString()).join(' '),
        question: input
      });

      return await this.model.invoke(finalPrompt);
    }
  }

  private reciprocalRankFusion(results: Documents[]): Documents {
    const k: number = 60;

    const fusedScores: Record<string, number> = {}
    const documents: Record<string, Document> = {}

    results.forEach(docs => {
      docs.forEach((doc, rank) => {
        // Use the document contents as the key for uniqueness
        const key = doc.pageContent;
        // If the document hasn't been seen yet,
        // - initialize score to 0
        // - save it for later
        if (!(key in fusedScores)) {
          fusedScores[key] = 0;
          documents[key] = doc;
        }
        // Update the score of the document using the RRF formula:
        // 1 / (rank + k)
        fusedScores[key] += 1 / (rank + k);
      })
    })

    // Sort the documents based on their fused scores in descending order 
    // to get the final re-ranked results
    const sorted = Object.entries(fusedScores).sort((a, b) => b[1] - a[1])
    // retrieve the corresponding doc for each key
    return sorted.map(([key]) => documents[key])

  }

  ragFusion(prompt: ChatPromptTemplate<PromptInputs>) {
    const perspectivePrompt = ChatPromptTemplate.fromTemplate(`
      You are a helpful assistant that generates multiple search queries based on a single input query.
      
      Generate multiple search queries related to: {question}

      Output (4 queries):
      `);

    const queries = perspectivePrompt.pipe(this.model).pipe(msgs => {
      return msgs.content.toString()
        .split('\n')
        .map(s => s.replaceAll('\\', '')).filter(s => !!s)
    });

    const retrievalChain = queries.pipe(msg => {
      return this.retriever.batch(msg)
    }).pipe(this.reciprocalRankFusion);

    return async (input: string) => {
      const docs = await retrievalChain.invoke({ question: input });
      const finalPrompt = await prompt.invoke({
        context: docs.map(d => d.pageContent.toString()).join(' '),
        question: input
      });

      this.model.temperature = 0;
      return await this.model.invoke(finalPrompt);
    }
    
  }

  hypotheticalDocumentEmbedding(prompt: ChatPromptTemplate<PromptInputs>) {
    const hydePrompt = ChatPromptTemplate.fromTemplate(`
      Please write a passage to answer the question
      Question: {question}
      Passage:`);

    this.model.temperature = 0;

    const retrievalChain = hydePrompt.pipe(this.model)
      .pipe(msg => msg.content)
      .pipe(this.retriever);
    
    return async (input: string) => {
      const docs = await retrievalChain.invoke({ question: input });
      const finalPrompt = await prompt.invoke({
        context: docs.map(d => d.pageContent.toString()).join(' '),
        question: input
      });

      return await this.model.invoke(finalPrompt);
    }
  }

  async callApi(question: string): Promise<ProviderResponse> {
    await this.initRetriever();

    const prompt = ChatPromptTemplate.fromTemplate(`
      Always answer the question from the provided context.

      Context: {context}

      Question: {question}
    `)

    const runner = RunnableLambda.from(
      // this.retrieveRewriteRead(prompt)
      // this.multiQueryRetrieval(prompt)
      // this.ragFusion(prompt)
      this.hypotheticalDocumentEmbedding(prompt)
    );


    const res = await runner.invoke(question);

    return {
      output: res.content
    }
  }
  
}

export async function queryTransformation() {
  
  const ins = new QueryTransformation({});

  const question = `
  Today I woke up and brushed my teeth, then I sat down to read
  the news. But then I forgot the food on the cooker.
  Who are the key figures in the ancient greek history of philosophy from Pre-Socratic period?
  `;
  
  const res = await ins.callApi(question);

  prettyPrint({ ans: res.output });

}