import { TextLoader } from 'langchain/document_loaders/fs/text';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { prettyPrint } from '../utils';


export async function docRecursiveTextSplitter() {

  const loader = new TextLoader('./files/test.txt');
  const docs = await loader.load();

  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 100,
    chunkOverlap: 20,
  });

  const docChunks = await splitter.splitDocuments(docs);

  prettyPrint({docChunks})

}

export async function pythonLangSplitter() {
  const PYTHON_CODE = `
    def hello_world():
      print("Hello, World!")

    # Call the function
    hello_world()
    `;
  
  const pySplitter = RecursiveCharacterTextSplitter.fromLanguage("python", {
    chunkSize: 10,
    chunkOverlap: 5,
  });

  const pyDocs = await pySplitter.createDocuments([PYTHON_CODE]);

  prettyPrint({ pyDocs });
}