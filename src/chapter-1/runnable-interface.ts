import { ProviderResponse } from 'promptfoo';
import { ChatPromptTemplate } from '@langchain/core/prompts';
import { Example } from '../shared/base-example-api-provider';
import { RunnableLambda } from '@langchain/core/runnables';


export default class RunnableInterfaceExample extends Example {
  async callApi(input: string): Promise<ProviderResponse> {
    const template = ChatPromptTemplate.fromMessages([
      ['system', `Answer the question based on the context below.`],
      ['human', 'Context: {context}'],
      ['human', 'Question: {question}']
    ]);

    const model = this.model;

    const runnableChatbot = RunnableLambda.from(async function* (values) {
      const prompt = await template.invoke(values);

      // return await this.model.invoke(prompt);

      // to enable Stream

      for await (const token of await model.stream(prompt)) {
        yield token
      }
    });

    const runnable = await runnableChatbot.stream({
      context: `
        The most recent advancements in NLP are being driven by Large 
        Language Models (LLMs). These models outperform their smaller 
        counterparts and have become invaluable for developers who are creating 
        applications with NLP capabilities. Developers can tap into these models 
        through Hugging Face's \`transformers\` library, or by utilizing OpenAI 
        and Cohere's offerings through the \`openai\` and \`cohere\` libraries, 
        respectively.
      `,
      question: input
    });

    let res:string[] = []

    for await (const token of runnable) {
      res.push((<any>token).content.toString())
   }

    return {
      output: res.join()
    }
  }
}