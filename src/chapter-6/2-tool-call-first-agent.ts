import { ProviderResponse } from 'promptfoo';
import { TavilySearchResults } from '@langchain/community/tools/tavily_search';
import { Calculator } from '@langchain/community/tools/calculator';
import { Annotation, messagesStateReducer, START, StateGraph } from '@langchain/langgraph';
import { AIMessage, BaseMessage, HumanMessage } from '@langchain/core/messages';

import { ToolNode, toolsCondition } from "@langchain/langgraph/prebuilt";

import { Example } from '../shared/base-example-api-provider';
import { prettyPrint } from '../utils';

// import getEnvConfig from '../shared/get-env-config';


export class ToolCallFirstAgent extends Example {

  async callApi(prompt: string): Promise<ProviderResponse> {
    const search = new TavilySearchResults({
      maxResults: 5,
      // apiKey: getEnvConfig.TAVILY_API_KEY [No need it will be automatically picked]
    });
    const calculator = new Calculator();

    const tools = [
        search,
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

    const toolCallFirstNode = async (state: typeof stateAnnotation.State) => {
      const query = state.messages[state.messages.length - 1].content;

      const searchToolCall = {
        name: search.name,
        args: { input: query },
        id: Math.random().toString(),
        type: 'tool_call' as 'tool_call'
      };

      return {
        messages:[
          new AIMessage({
            content: '',
            tool_calls: [searchToolCall]
          })]
      };
    };

    const modelNode = async (state: typeof stateAnnotation.State) => {
      const res = await this.model.invoke(state.messages);
      return {
        messages: res
      }
    }

    const toolNode = new ToolNode(tools);

    const graph = new StateGraph(stateAnnotation)
      .addNode('tool_call_first', toolCallFirstNode)
      .addNode('model', modelNode)
      .addNode('tools', toolNode)
      .addEdge(START, 'tool_call_first')
      .addEdge('tool_call_first', 'tools')
      .addEdge('tools', 'model')
      .addConditionalEdges('model', toolsCondition)
      .compile();
    
    const input = {
      messages: [
        new HumanMessage(prompt)
      ]
    };

    for await (const chunk of await graph.stream(input)) {
      prettyPrint(chunk);
    }

    return {};

    // const res = await graph.invoke(input);

    // return {
    //   output: res
    // };
  }
  
}


export async function toolCallFirstAgent() {
  const example = new ToolCallFirstAgent();

  const prompt = 'How old was the 30th president of the United States when he died?';

  const res = await example.callApi(prompt);

  prettyPrint(res);
}