import { TextLoader } from 'langchain/document_loaders/fs/text';
import { prettyPrint } from '../utils';

export async function textLoaderExample() {
  const loader = new TextLoader('./files/test.txt');
  
  const docs = await loader.load();

  prettyPrint({docs})
}