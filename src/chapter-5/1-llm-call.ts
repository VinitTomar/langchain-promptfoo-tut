import { ProviderResponse } from 'promptfoo';
import { Annotation, END, messagesStateReducer, START, StateGraph } from '@langchain/langgraph';

import { Example } from '../shared/base-example-api-provider';
import { prettyPrint } from '../utils';
import { BaseMessage, HumanMessage, SystemMessage } from '@langchain/core/messages';

export class LLMCallArch extends Example {

  private async llmCallGraph(userInput: string) {
    const stateAnnotation = Annotation.Root({
      messages: Annotation<BaseMessage[]>({
        reducer: messagesStateReducer,
        default: () => []
      })
    });

    const chatBotNode = async (state: typeof stateAnnotation.State) => {
      const answer = await this.model.invoke(state.messages);

      return {
        messages: answer
      }
    }

    const graph = new StateGraph(stateAnnotation)
      .addNode('chatbot', chatBotNode)
      .addEdge('chatbot', END)
      .addEdge(START, 'chatbot')
      .compile();
    
    const input = {
      messages: [
        new HumanMessage(userInput)
      ]
    };

    const res = await graph.invoke(input);

    return res.messages;
  }

  async callApi(userQuery: string): Promise<ProviderResponse> {
    return {
      output: await this.llmCallGraph(userQuery)
    };
  }
}

export async function llmCallArchExample() {
  const memSys = new LLMCallArch({});
  
  const userQuery = 'Hi!';
  const res = await memSys.callApi(userQuery);

  prettyPrint(res);
}