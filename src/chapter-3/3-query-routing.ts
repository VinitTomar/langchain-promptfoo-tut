import { ProviderResponse } from 'promptfoo';
import { z } from 'zod';
import { ChatPromptTemplate, PromptTemplate } from '@langchain/core/prompts';

import { Example } from '../shared/base-example-api-provider';
import { prettyPrint } from '../utils';
import { GeminiEmbeddings } from '../shared/gemini-embeddings';
import { cosineSimilarity } from 'langchain/util/math';


export class QueryRouting extends Example {

  private async logicalRouting(userQuery: string) {
    const routeQueryStructure = z.object({
      dataSource: z.enum(['python_docs', 'js_docs']).describe('Given a user\'s question choose the datasource which will be most relevant to the user\'s question')
    }).describe('Route a user\s query to the most relevant datasource');

    this.model.temperature = 0;

    const structuredLLM = this.model.withStructuredOutput(routeQueryStructure, {
      name: 'routeQuery'
    });

    const prompt = ChatPromptTemplate.fromMessages([
      ['system', 'You are an expert routing user\'s question to the right datasource. Based on the programming language question is referring, route it to correct data source.'],
      ['human', '{question}']
    ]);

    const router = prompt.pipe(structuredLLM);

    return await router.invoke({question: userQuery});

  }

  private async semanticRouting(userQuery: string) {
    const physicsTemplate = `You are a very smart physics professor. You are great at answering questions about physics in a concise and easy-to-understand manner. When you don't know the answer to a question, you admit that you don't know.

    Here is a question:
    {query}`;

    const mathTemplate = `You are a very good mathematician. You are great at answering math questions. You are so good because you are able to break down hard problems into their component parts, answer the component parts, and then put them together to answer the broader question.

    Here is a question:
    {query}`;

    const embedding = new GeminiEmbeddings();
    const promptTemplates = [physicsTemplate, mathTemplate];
    const templateEmbeddings = await embedding.embedDocuments(promptTemplates);
    const queryEmbedding = await embedding.embedQuery(userQuery);
    const similarities = cosineSimilarity([queryEmbedding], templateEmbeddings)[0];

    const similarPrompt = similarities[0] > similarities[1]
      ? promptTemplates[0] : promptTemplates[1];

    const finalPrompt = PromptTemplate.fromTemplate<{ query: string }>(similarPrompt);

    const res = await this.model.invoke(await finalPrompt.invoke({ query: userQuery }));

    return res.content;

  }
  
  async callApi(userQuery: string): Promise<ProviderResponse> {
   
    return {
      // output: await this.logicalRouting(userQuery)
      output: await this.semanticRouting(userQuery)
    }

  }
}

export async function queryRouting() {
  
  const ins = new QueryRouting({});

  // const question = `
  //   Why doesn't the following code work:

  //   import { ChatPromptTemplate } from '@langchain/core/prompts';

  //   prompt = ChatPromptTemplate.fromMessages(["human", "speak in {language}"])
  //   await prompt.invoke("french")
  // `;

  const question = "What is a black hole?";
  
  const res = await ins.callApi(question);

  prettyPrint({ ans: res.output });

}