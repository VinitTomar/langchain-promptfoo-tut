import { ProviderResponse } from 'promptfoo';
import { Document } from 'langchain/document';
import { AttributeInfo } from 'langchain/chains/query_constructor';

import { MemoryVectorStore } from 'langchain/vectorstores/memory';
import { SelfQueryRetriever } from 'langchain/retrievers/self_query';
import { FunctionalTranslator } from '@langchain/core/structured_query';
import { DataSource } from 'typeorm';
import { SqlDatabase } from 'langchain/sql_db';

import { Example } from '../shared/base-example-api-provider';
import { prettyPrint } from '../utils';
import { GeminiEmbeddings } from '../shared/gemini-embeddings';
import { createSqlQueryChain } from 'langchain/chains/sql_db';
import { QuerySqlTool } from 'langchain/tools/sql';
import { ChatPromptTemplate } from '@langchain/core/prompts';

export class QueryConstruction extends Example {

  private async textToMetadata(userQuery: string) {
    const docs = [
      new Document({
        pageContent:
          'A bunch of scientists bring back dinosaurs and mayhem breaks loose',
        metadata: {
          year: 1993,
          rating: 7.7,
          genre: 'science fiction',
          length: 122,
        },
      }),
      new Document({
        pageContent:
          'Leo DiCaprio gets lost in a dream within a dream within a dream within a ...',
        metadata: {
          year: 2010,
          director: 'Christopher Nolan',
          rating: 8.2,
          length: 148,
        },
      }),
      new Document({
        pageContent:
          'A psychologist / detective gets lost in a series of dreams within dreams within dreams and Inception reused the idea',
        metadata: { year: 2006, director: 'Satoshi Kon', rating: 8.6 },
      }),
      new Document({
        pageContent:
          'A bunch of normal-sized women are supremely wholesome and some men pine after them',
        metadata: {
          year: 2019,
          director: 'Greta Gerwig',
          rating: 8.3,
          length: 135,
        },
      }),
      new Document({
        pageContent: 'Toys come alive and have a blast doing so',
        metadata: { year: 1995, genre: 'animated', length: 77 },
      }),
      new Document({
        pageContent: 'Three men walk into the Zone, three men walk out of the Zone',
        metadata: {
          year: 1979,
          director: 'Andrei Tarkovsky',
          genre: 'science fiction',
          rating: 9.9,
        },
      }),
    ];
    
    const fields = [
      {
        name: 'genre',
        description: 'The genre of the movie',
        type: 'string or array of strings',
      },
      {
        name: 'year',
        description: 'The year the movie was released',
        type: 'number',
      },
      {
        name: 'director',
        description: 'The director of the movie',
        type: 'string',
      },
      {
        name: 'rating',
        description: 'The rating of the movie (1-10)',
        type: 'number',
      },
      {
        name: 'length',
        description: 'The length of the movie in minutes',
        type: 'number',
      },
    ];

    const attributeInfo = fields.map(
      (field) => new AttributeInfo(field.name, field.description, field.type)
    );

    this.model.temperature = 0;
    const embeddings = new GeminiEmbeddings();
    const vectorStore = await MemoryVectorStore.fromDocuments(docs, embeddings);

    const selfQueryRetriever = SelfQueryRetriever.fromLLM({
      llm: this.model,
      vectorStore,
      documentContents: 'Brief summary of a movie',
      attributeInfo,
      structuredQueryTranslator: new FunctionalTranslator()
    })

    return await selfQueryRetriever.invoke(userQuery);
  }

  private async textToSQL(userQuery: string) {
    const dbDialect = 'sqlite';
    const dataSource = new DataSource({
      type: dbDialect,
      database: './files/chinook.db',
    });

    const db = await SqlDatabase.fromDataSourceParams({
      appDataSource: dataSource
    });

    this.model.temperature = 0;

    const sqlQueryChain = await createSqlQueryChain({
      llm: this.model,
      db,
      dialect: dbDialect,
    });

    const sqlQueryExecuteTool = new QuerySqlTool(db);

    const sqlQuery = await sqlQueryChain.invoke({ question: userQuery });
    const result = await sqlQueryExecuteTool.invoke(
      sqlQuery.replaceAll('```', '').replaceAll(dbDialect, '')
    );
    return result;
    
  }

  
  async callApi(userQuery: string): Promise<ProviderResponse> {
    return {
      // output: await this.textToMetadata(userQuery)
      output: await this.textToSQL(userQuery)
      
    }
  }
}

export async function queryConstruction() {
  
  const ins = new QueryConstruction({});

  // const question = "Which movies are rated higher than 8.5?";
  const question = 'How many employees are there?';
  
  const res = await ins.callApi(question);

  prettyPrint({ ans: res.output });

}