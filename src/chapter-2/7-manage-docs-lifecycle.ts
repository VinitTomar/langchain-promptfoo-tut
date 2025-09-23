
import { index } from 'langchain/indexes';
import { v4 as uuidv4 } from 'uuid';
import { PgDbVectorStore } from '../shared/pg-db-vector-store';
import { Documents, prettyPrint } from '../utils';

export async function manageDocsLifeCycle() {
  const vsDb = await PgDbVectorStore.getVectorStore('mgt_doc_lif_cycle');

  const recordManager = await PgDbVectorStore.getPgRecordManager(
    'record_manager_tbl', 'record_manager_nm'
  );

  const docs: Documents = [
    {
      pageContent: 'There are cats in the pond.',
      metadata: {
        id: uuidv4(),
        source: 'cats.txt'
      }
    },

    {
      pageContent: 'Ducks are found in the pond.',
      metadata: {
        id: uuidv4(),
        source: 'ducks.txt'
      }
    }
  ];


  const indexAttempt1 = await index({
    docsSource: docs,
    recordManager,
    vectorStore: vsDb,
    options: {
      cleanup: 'incremental',
      sourceIdKey: 'source'
    }
  });

  prettyPrint({ indexAttempt1 });

  const indexAttempt2 = await index({
    docsSource: docs,
    recordManager,
    vectorStore: vsDb,
    options: {
      cleanup: 'incremental',
      sourceIdKey: 'source'
    }
  });

  prettyPrint({ indexAttempt2 });


  docs[0].pageContent += 'Content modified';

  const indexAttempt3 = await index({
    docsSource: docs,
    recordManager,
    vectorStore: vsDb,
    options: {
      cleanup: 'incremental',
      sourceIdKey: 'source'
    }
  });

  prettyPrint({ indexAttempt3 });
}