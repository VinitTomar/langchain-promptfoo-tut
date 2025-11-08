import { AIMessage, HumanMessage } from '@langchain/core/messages';
import { Annotation, CompiledStateGraph, END, messagesStateReducer, START, StateGraph, StreamMode } from '@langchain/langgraph';
import { Example } from '../shared/base-example-api-provider';
import { ProviderResponse } from 'promptfoo';
import { prettyPrint } from '../utils';


class IntermediateOutput extends Example {


  private async steamWithMode(mode: StreamMode, streamEvent = false) {
    const stateAnnotation = Annotation.Root({
      messages: Annotation({
        reducer: messagesStateReducer,
        default: () => []
      })
    });

    const firstNode = async (state: typeof stateAnnotation.State) => {
      const msg = state.messages[0];

      return {
        messages: new AIMessage(msg ? msg.content.toString() : 'No first msg found') 
      }
    };

    const secondNode = async (state: typeof stateAnnotation.State) => {
      const msg = state.messages[1];

      return {
        messages: new AIMessage(msg ? msg.content.toString() : 'No second msg found')
      }
    };

    const graph = new StateGraph(stateAnnotation)
      .addNode('first', firstNode)
      .addNode('second', secondNode)
      .addEdge(START, 'first')
      .addEdge('first', 'second')
      .addEdge('second', END)
      .compile();
    
    const input = {
      messages: [
        // new HumanMessage('Input from human')
      ]
    };

    if (streamEvent) {
      const output = graph.streamEvents(input, {
        version: 'v2'
      });

      for await (const o of output) {
        prettyPrint(o)
      }

    } else {
      const output = await graph.stream(input, {
        streamMode: mode
      });
  
      for await (const c of output) {
        console.log(c);
      }

    }

  }

  async callApi(prompt: string): Promise<ProviderResponse> {
    await this.steamWithMode('values', true);
    return  {};
  }

}

export async function intermediateOutput() {
  const example = new IntermediateOutput();
  
    const prompt = '';
  
    const res = await example.callApi(prompt);
  
    prettyPrint(res);
}