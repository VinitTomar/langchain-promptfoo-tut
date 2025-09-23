import { TextLoader } from 'langchain/document_loaders/fs/text';
import { RecursiveCharacterTextSplitter, RecursiveCharacterTextSplitterParams } from 'langchain/text_splitter';


export class TextLoaderWithRecursiveSplitter {
 
  constructor(
    private filePath: string,
    private splitterConfig: Partial<RecursiveCharacterTextSplitterParams> = {
      chunkSize: 1000,
      chunkOverlap: 200,
    }
  ){}

  async loadSplittedDocs() {
    const loader = new TextLoader(this.filePath);
    const rawDocs = await loader.load();
    const splitter = new RecursiveCharacterTextSplitter(this.splitterConfig);
    return splitter.splitDocuments(rawDocs);
  }

}