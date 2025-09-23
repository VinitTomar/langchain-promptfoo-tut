

import { PDFLoader } from '@langchain/community/document_loaders/fs/pdf';
import { prettyPrint } from '../utils';

export async function pdfDocsLoader() {
  const loader = new PDFLoader("./files/test.pdf");

  const docs = await loader.load();

  prettyPrint({docs})

}
