import { PGVectorStore } from '@langchain/community/vectorstores/pgvector';
import { PostgresRecordManager } from '@langchain/community/indexes/postgres';

import { GeminiEmbeddings } from '../shared/gemini-embeddings';
import { Documents } from '../utils';
import { connectionUrl } from './pg-db-conn';


export class PgDbVectorStore {
  private static embeddings_model = new GeminiEmbeddings();

  private static db: PGVectorStore;
  private static pgRecordManager: PostgresRecordManager;
  private static readonly dbTableName = 'langchain_tut';

  private static readonly connectionString = connectionUrl()

  private static readonly recordTableName = 'upserting_records';
  private static readonly recordNameSpace = 'test_namespace';

  private static getTableName(tableName: string): string {
    /**
     * A hack to not manually delete table to avoid adding duplicate vector entries.
     * NOT FOR PRODUCTION.
     */
    return `${tableName}_${Date.now().toString()}`
  }
  
  static async getVectorStore(tableName?: string) {
    if (this.db)
      return this.db;

    this.db = await PGVectorStore.initialize(this.embeddings_model, {
      postgresConnectionOptions: {
        connectionString: this.connectionString
      },
      tableName: this.getTableName(tableName || this.dbTableName)
    });

    return this.db;
  }

  static async getVectorStoreAsRetriever(
    { docs, tableName, k }: { tableName?: string, docs?: Documents, k?: number }
  ) {
    if (docs) {
      return (await this.fromDocuments(docs, tableName)).asRetriever({k});
    }

    return (await this.getVectorStore(tableName)).asRetriever({k});
  }

  static async fromDocuments(docs: Documents, tableName?: string) {
    if (this.db)
      return this.db;

    this.db = await PGVectorStore.fromDocuments(docs, this.embeddings_model, {
      postgresConnectionOptions: {
        connectionString: this.connectionString
      },
      tableName: this.getTableName(tableName || this.dbTableName)
    });

    return this.db; 
  }

  static async getPgRecordManager(
    tableName?: string, namespace?: string
  ) {
    if (this.pgRecordManager)
      return this.pgRecordManager;

    this.pgRecordManager = new PostgresRecordManager(
      namespace || this.recordNameSpace,
      {
        postgresConnectionOptions: {
          connectionString: this.connectionString
        },
        tableName: tableName || this.recordTableName
      }
    )

    await this.pgRecordManager.createSchema();

    return this.pgRecordManager;
  }
}