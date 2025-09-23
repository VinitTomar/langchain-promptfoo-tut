import { ChatPromptTemplate as LcChatPromptTemplate} from "@langchain/core/prompts";
import { ProviderResponse } from 'promptfoo';
import { Example } from '../shared/base-example-api-provider';


export default class ChatPromptTemplate extends Example {

   /** 
     * TODO: need to check again because Gemini is not returning Hugging face in answer.
    */
  
  async callApi(question: string): Promise<ProviderResponse> {

    const promptTemplate = LcChatPromptTemplate.fromMessages([
      ['system', `Answer the question based on the context below. If the question 
    cannot be answered using the information provided, answer with "I 
    don\'t know".`],
      ['human', 'Context: {context}'],
      ['human', 'Question: {question}']
    ]);

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