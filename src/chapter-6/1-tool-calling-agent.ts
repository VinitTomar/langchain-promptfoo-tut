import { ProviderResponse } from 'promptfoo';
import { DuckDuckGoSearch } from '@langchain/community/tools/duckduckgo_search';
import { Calculator } from '@langchain/community/tools/calculator';
import { Annotation, messagesStateReducer, START, StateGraph } from '@langchain/langgraph';
import { BaseMessage, HumanMessage } from '@langchain/core/messages';

import { ToolNode, toolsCondition } from "@langchain/langgraph/prebuilt";

import { Example } from '../shared/base-example-api-provider';
import { prettyPrint } from '../utils';
import { tool } from '@langchain/core/tools';


export class ToolCallingAgent extends Example {

  async callApi(prompt: string): Promise<ProviderResponse> {
    const search = new DuckDuckGoSearch();
    const calculator = new Calculator();

    /**
     * Tools are not called. Need to find a better example.
     */

    const tools = [
        tool(async (input) => {
          console.log("========== Calling DDG search tool ==========");
          prettyPrint({ searchInput: input });

          const modelGeneratedToolCall = {
            args: {
              input,
            },
            id: "tool_call_id",
            name: search.name,
            type: "tool_call",
          };

          return await search.invoke(modelGeneratedToolCall);
        }, { ...search }),
        
        calculator
      ];

    this.model.temperature = 0;
    this.model.bindTools(tools);

    const stateAnnotation = Annotation.Root({
      messages: Annotation<BaseMessage[]>({
        reducer: messagesStateReducer,
        default: () => [],
      }),
    });

    const modelNode = async (state: typeof stateAnnotation.State) => {
      const res = await this.model.invoke(state.messages);
      return {
        messages: res
      }
    }
    const toolNode = new ToolNode(tools);

    const graph = new StateGraph(stateAnnotation)
      .addNode('model', modelNode)
      .addNode('tools', toolNode)
      .addEdge(START, 'model')
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


export async function toolCallingAgent() {
  const example = new ToolCallingAgent();

  const prompt = 'How old was the 30th president of the United States when he died?';

  const res = await example.callApi(prompt);

  prettyPrint(res);
}