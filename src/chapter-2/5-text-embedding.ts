import { TextLoader } from 'langchain/document_loaders/fs/text';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { GeminiEmbeddings, prettyPrint } from '../utils';


export async function textEmbedding() {
  const filePath = './files/test.txt';

  const loader = new TextLoader(filePath);

  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 100,
    chunkOverlap: 20,
  });

  const embedding = new GeminiEmbeddings();
  
  const chunks = await splitter.splitDocuments(
    await loader.load()
  );

  const embed = await embedding.embedDocuments(
    chunks.map(c => c.pageContent)
  );

  prettyPrint({embed})
}