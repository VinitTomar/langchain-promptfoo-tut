import { 
  CheerioWebBaseLoader 
} from "@langchain/community/document_loaders/web/cheerio";
import { prettyPrint } from '../utils';

export async function webBaseLoader() {
  const loader = new CheerioWebBaseLoader('https://www.langchain.com/');

  const docs = await loader.load();

  prettyPrint({docs})
}