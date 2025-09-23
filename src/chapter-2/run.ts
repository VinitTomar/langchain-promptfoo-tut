import { textLoaderExample } from './1-text-loader';
import { pdfDocsLoader } from './3-pdf-loader';
import { webBaseLoader } from './2-web-base-loader';
import { docRecursiveTextSplitter, pythonLangSplitter } from './4-recursive-text-splitter';
import { textEmbedding } from './5-text-embedding';
import { storeEmbeddingPgVector } from './6-store-embedding-pg-vector';
import { manageDocsLifeCycle } from './7-manage-docs-lifecycle';


(async () => {
  
  // await textLoaderExample();
  // await webBaseLoader();
  // await pdfDocsLoader();

  // await docRecursiveTextSplitter();
  // await pythonLangSplitter();

  // await textEmbedding();
  // await storeEmbeddingPgVector();
  await manageDocsLifeCycle();

})();