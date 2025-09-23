import { ProviderResponse } from 'promptfoo';
import * as z from 'zod';

import { Example } from '../shared/base-example-api-provider';

export default class SpecificOutputFormat extends Example {
  async callApi(prompt: string): Promise<ProviderResponse> {
    const answerSchema = z.object({
      answer: z.string().describe('The answer to the user\'s question'),
      justification: z.string().describe('Justification to the answer')
    }).describe('An answer to the user\'s question with justification for the answer');

    const res = await this.model.withStructuredOutput(answerSchema).invoke(prompt);

    return {
      output: res
    }

  }
}