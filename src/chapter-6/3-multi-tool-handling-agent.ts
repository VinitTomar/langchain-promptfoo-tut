import { ProviderResponse } from 'promptfoo';
import { DuckDuckGoSearch } from '@langchain/community/tools/duckduckgo_search';
import { Calculator } from '@langchain/community/tools/calculator';
import { Annotation, messagesStateReducer, START, StateGraph } from '@langchain/langgraph';
import { BaseMessage, HumanMessage } from '@langchain/core/messages';

import { ToolNode, toolsCondition } from "@langchain/langgraph/prebuilt";

import { Example, ExampleWithEmbeddings } from '../shared/base-example-api-provider';
import { prettyPrint } from '../utils';
import { tool } from '@langchain/core/tools';
import { TavilySearchResults } from '@langchain/community/tools/tavily_search';
import { MemoryVectorStore } from 'langchain/vectorstores/memory';
import { Document } from "@langchain/core/documents";



export class MultiToolHandlingAgent extends ExampleWithEmbeddings {

  async callApi(prompt: string): Promise<ProviderResponse> {
    const search = new TavilySearchResults({ maxResults: 1 });
    const calculator = new Calculator();
    
    const tools = [
      search,
      calculator
    ];

    const toolsStore = await MemoryVectorStore.fromDocuments(
      tools.map(tool => {
        return new Document({
          pageContent: tool.description,
          metadata: {
            name: tool.name
          }
        })
      }),
      this.embeddings
    );
    const toolsRetriever = toolsStore.asRetriever();
    
    this.model.temperature = 0;
    
    const stateAnnotation = Annotation.Root({
      messages: Annotation({
        reducer: messagesStateReducer,
        default: () => []
      }),
      selectedTools: Annotation<string[]>(),
    });

    const selectToolsNode = async (state: typeof stateAnnotation.State) => {
      const query = state.messages[state.messages.length - 1].content;
      const toolDocs = await toolsRetriever.invoke(query.toString());

      return {
        selectedTools: toolDocs.map(doc => doc.metadata.name)
      }
    }

    const modelNode = async (state: typeof stateAnnotation.State) => {
      const selectedTools = tools.filter(tool => {
        return state.selectedTools.includes(tool.name)
      });

      this.model.bindTools(selectedTools);

      const res = await this.model.invoke(state.messages);

      return {
        messages: res
      }
    }

    const graph = new StateGraph(stateAnnotation)
      .addNode('select_tool', selectToolsNode)
      .addNode('model', modelNode)
      .addNode('tools', new ToolNode(tools))
      .addEdge(START, 'select_tool')
      .addEdge('select_tool', 'model')
      .addConditionalEdges('model', toolsCondition)
      .addEdge('tools', 'model')
      .compile();
    
    const input = {
      messages: [
        new HumanMessage(prompt)
      ]
    };

    for await (const chunk of await graph.stream(input)) {
      prettyPrint(chunk);
    }

    return {}

    // const res = await graph.invoke(input);

    // return {
    //   output: res
    // };
  }
  
}


export async function multiToolHandlingAgent() {
  const example = new MultiToolHandlingAgent();

  const prompt = 'How old was the 30th president of the United States when he died?';

  const res = await example.callApi(prompt);

  prettyPrint(res);
}