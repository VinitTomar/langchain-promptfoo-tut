import { GoogleGenerativeAIEmbeddings } from '@langchain/google-genai';
import path from 'path';
import dotenv from 'dotenv';


dotenv.config({
  path: path.join(__dirname, '../.env')
});

export class GeminiEmbeddings extends GoogleGenerativeAIEmbeddings{
  constructor(modelName: string = 'gemini-embedding-001') {
    super({
      model: modelName,
      apiKey: process.env.GOOGLE_API_KEY
    })
  }
}