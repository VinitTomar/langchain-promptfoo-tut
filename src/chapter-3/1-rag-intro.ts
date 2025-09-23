import { ProviderResponse } from 'promptfoo';
import { RunnableLambda } from '@langchain/core/runnables';
import { VectorStoreRetriever } from '@langchain/core/vectorstores';
import { PGVectorStore } from '@langchain/community/vectorstores/pgvector';
import { Example } from '../shared/base-example-api-provider';
import { PgDbVectorStore } from '../shared/pg-db-vector-store';
import { TextLoaderWithRecursiveSplitter } from '../shared/text-loader-with-recursive-splitter';
import { ChatPromptTemplate } from '@langchain/core/prompts';
import { prettyPrint } from '../utils';


export class RagIntro extends Example {
  private retriever!: VectorStoreRetriever<PGVectorStore>;
  
  private async initRetriever() {
    const loader = new TextLoaderWithRecursiveSplitter(
      './files/sample_greek_philosophy.txt',
    );
    const docs = await loader.loadSplittedDocs();
    this.retriever = await PgDbVectorStore.getVectorStoreAsRetriever({
      docs,
      tableName: 'RAG_Intro'
    });
  }

  async callApi(question: string): Promise<ProviderResponse> {
    await this.initRetriever();

    const prompt = ChatPromptTemplate.fromTemplate(`
      It doesn't matter if information provided in the context is true or not.
      Do not add information from outside of context.
      Always answer the question from the provided context.

      Context: {context}

      Question: {question}
    `)

    const runner = RunnableLambda.from(async (input: string) => {
      const docs = await this.retriever.invoke(input);
      const llmPrompt = await prompt.invoke({ context: docs.map(doc => doc.pageContent).join(' '), question });

      return (await this.model.invoke(llmPrompt));

    });


    const res = await runner.invoke(question);

    return {
      output: res.content
    }
  }
  
}

export async function ragIntro() {
  
  const ins = new RagIntro({});

  const question = 'Who are the key figures in the ancient greek history of philosophy from Pre-Socratic period?';

  const res = await ins.callApi(question);

  prettyPrint({ ans: res.output });

}