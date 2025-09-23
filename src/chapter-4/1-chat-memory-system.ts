import { ProviderResponse } from 'promptfoo';
import { Annotation, END, MemorySaver, messagesStateReducer, START, StateGraph } from '@langchain/langgraph';

import { Example } from '../shared/base-example-api-provider';
import { prettyPrint } from '../utils';
import { BaseMessage, HumanMessage } from '@langchain/core/messages';

export class ChatMemorySystem extends Example {
  
  private async simpleGraph(userQuery: string) {
     const stateAnnotation = Annotation.Root({
      messages: Annotation<BaseMessage[]>({
        reducer: messagesStateReducer,
        default: () => []
      })
     });
    
    const chatBoatNode = async (state: typeof stateAnnotation.State) => {
      const answer = await this.model.invoke(state.messages);
      return { messages: answer };
    }

    const graph = new StateGraph(stateAnnotation)
      .addNode('chat_boat', chatBoatNode)
      .addEdge(START, 'chat_boat')
      .addEdge('chat_boat', END)
      .compile();
    

    const input = {
      messages: [
        new HumanMessage(userQuery)
      ]
    };

    const res = await graph.invoke(input);

    return res.messages;
  }
  
  private async simpleGraphWithMemory() {
     const stateAnnotation = Annotation.Root({
      messages: Annotation<BaseMessage[]>({
        reducer: messagesStateReducer,
        default: () => []
      })
     });
    
    const chatBoatNode = async (state: typeof stateAnnotation.State) => {
      const answer = await this.model.invoke(state.messages);
      return { messages: answer };
    }

    const graph = new StateGraph(stateAnnotation)
      .addNode('chat_boat', chatBoatNode)
      .addEdge(START, 'chat_boat')
      .addEdge('chat_boat', END)
      .compile({
        checkpointer: new MemorySaver()
      });
    

    const thread1 = {
      configurable: {
        thread_id: 1
      }
    };

    const res1: Record<string, BaseMessage[]> = {
      result1: [],
      result2: []
    }

    const input1 = {
      messages: [
        new HumanMessage('Hi, my name is Vinit.')
      ]
    };

    res1.result1 = (await graph.invoke(
      input1,
      thread1
    )).messages;

    const input2 = {
      messages: [
        new HumanMessage('what is my name?')
      ]
    };

    res1.result2 = (await graph.invoke(
      input2,
      thread1
    )).messages;

    return res1;
  }

  async callApi(userQuery: string): Promise<ProviderResponse> {
    return {
      output: await this.simpleGraphWithMemory()
    };
  }
}

export async function chatMemorySystem() {
  const memSys = new ChatMemorySystem({});
  const userQuery = 'Hi!';

  const res = await memSys.callApi(userQuery);

  prettyPrint(res);
}