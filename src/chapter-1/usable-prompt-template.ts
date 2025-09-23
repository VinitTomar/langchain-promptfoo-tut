import { PromptTemplate } from "@langchain/core/prompts";
import { ProviderResponse } from 'promptfoo';
import { Example } from '../shared/base-example-api-provider';


export default class UsablePromptTemplate extends Example {
  
  async callApi(question: string): Promise<ProviderResponse> {

    /** 
     * TODO: need to check again because Gemini is not returning Hugging face in answer.
    */

    const promptTemplate = PromptTemplate.fromTemplate(`
      Answer the question based on the context below. If question can not be answered based on the information, reply "I don't know".

      Context: {context}

      Question: {question}

      Answer:
    `);

    const prompt = await promptTemplate.invoke({
      context: `
        The most recent advancements in NLP are being driven by Large 
        Language Models (LLMs). These models outperform their smaller 
        counterparts and have become invaluable for developers who are creating 
        applications with NLP capabilities. Developers can tap into these models 
        through Hugging Face's \`transformers\` library, or by utilizing OpenAI 
        and Cohere's offerings through the \`openai\` and \`cohere\` libraries, 
        respectively.
      `,
      question
    })

    const res = await this.model.invoke(prompt);

    return {
      output: res.content
    }
  }
}