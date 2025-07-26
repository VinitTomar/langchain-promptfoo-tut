import { ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings } from '@langchain/google-genai';
import { ApiProvider, ProviderOptions, ProviderResponse } from 'promptfoo';
import { Document } from '@langchain/core/documents';
import dotenv from 'dotenv';
import path from 'path';
import { color, colorize } from 'json-colorizer';

dotenv.config({
  path: path.join(__dirname, '../.env')
});

export type Documents = Document[];

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

export function prettyPrint(obj: any) {

  const defaultTheme = {
    Whitespace: color.gray,
    Brace: color.gray,
    Bracket: color.gray,
    Colon: color.gray,
    Comma: color.gray,
    StringKey: color.magenta,
    StringLiteral: color.yellow,
    NumberLiteral: color.green,
    BooleanLiteral: color.cyan,
    NullLiteral: color.white
};

  console.log(colorize(
    obj,
    {
      colors: {
        ...defaultTheme,
        StringKey: color.green,
        StringLiteral: color.white,
      }
    }
  ));
}

export class GeminiEmbeddings extends GoogleGenerativeAIEmbeddings{
  constructor(modelName: string = 'gemini-embedding-001') {
    super({
      model: modelName,
      apiKey: process.env.GOOGLE_API_KEY
    })
  }
}