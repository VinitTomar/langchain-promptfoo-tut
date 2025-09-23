import { TextLoader } from 'langchain/document_loaders/fs/text';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { prettyPrint } from '../utils';
import { PgDbVectorStore } from '../shared/pg-db-vector-store';


export async function storeEmbeddingPgVector() {
  const loader = new TextLoader("./files/test.txt");
  const raw_docs = await loader.load();
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 100,
    chunkOverlap: 20,
  });
  const docs = await splitter.splitDocuments(raw_docs)

  const db = await PgDbVectorStore.getVectorStore('tmp_tut');

  await db.addDocuments(docs)

  const retrievedDbDocs = await db.similaritySearch('LangSmith', 3);

  prettyPrint({retrievedDbDocs})
}