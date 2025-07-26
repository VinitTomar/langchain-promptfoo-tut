import { ProviderResponse } from 'promptfoo';
import { Example } from '../utils';

export default class Example1 extends Example {

  async callApi(prompt: string): Promise<ProviderResponse> {

    const res = await this.model.invoke(prompt)

    return {
      output: res.content
    }
  }
  
}