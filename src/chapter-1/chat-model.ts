import { HumanMessage, SystemMessage } from '@langchain/core/messages';
import { ProviderResponse } from 'promptfoo';
import { Example } from '../shared/base-example-api-provider';


export default class ExampleChatModel extends Example {
  
  async callApi(question: string): Promise<ProviderResponse> {
    const prompt = [
      new SystemMessage("You are a helpful that answers to a question with four exclamation marks"),
      new HumanMessage(question)
    ];

    const res = await this.model.invoke(prompt);

    return {
      output: res.content
    }
  }
}