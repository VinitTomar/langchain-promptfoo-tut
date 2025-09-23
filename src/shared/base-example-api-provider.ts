import { ChatGoogleGenerativeAI } from '@langchain/google-genai';
import path from 'path';
import { ApiProvider, ProviderOptions, ProviderResponse } from 'promptfoo';
import dotenv from 'dotenv';


dotenv.config({
  path: path.join(__dirname, '../../.env')
});

export abstract class Example implements ApiProvider {
  protected model: ChatGoogleGenerativeAI;

  constructor(
    private options: ProviderOptions,
    modelName: string = "gemini-2.0-flash"
  ) {
    this.model = new ChatGoogleGenerativeAI({
      model: modelName,
      apiKey: process.env.GOOGLE_API_KEY
    });
  }

  id(): string {
    return this.options.id || "Simple custom provider";
  }

  abstract callApi(prompt: string): Promise<ProviderResponse>;
}