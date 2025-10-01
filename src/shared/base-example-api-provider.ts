import { ChatGoogleGenerativeAI } from '@langchain/google-genai';
import { ApiProvider, ProviderOptions, ProviderResponse } from 'promptfoo';
import { GeminiEmbeddings } from './gemini-embeddings';
import config from './get-env-config';


export abstract class Example implements ApiProvider {
  protected model: ChatGoogleGenerativeAI;

  constructor(
    protected options: ProviderOptions = {},
    modelName: string = "gemini-2.0-flash"
  ) {
    this.model = new ChatGoogleGenerativeAI({
      model: modelName,
      apiKey: config.GOOGLE_API_KEY
    });
  }

  id(): string {
    return this.options.id || "Simple custom provider";
  }

  abstract callApi(prompt: string): Promise<ProviderResponse>;
}

export abstract class ExampleWithEmbeddings extends Example {
  protected embeddings: GeminiEmbeddings;

  constructor(options?: ProviderOptions, modelName?: string) {
    super(options, modelName);
    this.embeddings = new GeminiEmbeddings();
  }

  id(): string {
    return this.options.id || "Simple custom provider with embeddings";
  }

  abstract callApi(prompt: string): Promise<ProviderResponse>;
}