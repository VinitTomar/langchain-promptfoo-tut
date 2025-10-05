import { ProviderResponse } from 'promptfoo';
import { Annotation, END, messagesStateReducer, START, StateGraph } from '@langchain/langgraph';
import { AIMessage, HumanMessage, SystemMessage } from '@langchain/core/messages';

import { Example } from '../shared/base-example-api-provider';
import { prettyPrint } from '../utils';



export class SubgraphExample extends Example {

  private async directCall(prompt: string) {
    const parentStateAnnotation = Annotation.Root({
      foo: Annotation()
    });

    const childStateAnnotation = Annotation.Root({
      foo: Annotation,
      bar: Annotation(),
    });

    const subgraphNode = async (state: typeof childStateAnnotation.State) => {
      return {
        foo: state.foo + 'bar'
      };
    }

    const subgraph = new StateGraph(childStateAnnotation)
      .addNode('child', subgraphNode)
      .addEdge(START, 'child')
      .addEdge('child', END)
      .compile();

    const graph = new StateGraph(parentStateAnnotation)
      .addNode('subgraph', subgraph)
      .addEdge(START, 'subgraph')
      .compile();
    
    const input = {
      foo: 'foo => '
    };

    for await (const chunk of await graph.stream(input)) {
      prettyPrint(chunk);
    }

    return {}
  }

  private async indirectCall(prompt: string) {
    const parentStateAnnotation = Annotation.Root({
      foo: Annotation()
    });

    const childStateAnnotation = Annotation.Root({
      bar: Annotation()
    });


    const childNode = async (state: typeof childStateAnnotation.State) => {
      return {
        bar: state.bar + " bar"
      }
    };

    const childGraph = new StateGraph(childStateAnnotation)
      .addNode('child', childNode)
      .addEdge(START, 'child')
      .addEdge('child', END)
      .compile();
    
    const childGraphWrapper = async (state: typeof parentStateAnnotation.State) => {
      const res = await childGraph.invoke({
        bar: state.foo
      });

      return {
        foo: res.bar
      }
    };

    const graph = new StateGraph(parentStateAnnotation)
      .addNode('subgraph', childGraphWrapper)
      .addEdge(START, 'subgraph')
      .addEdge('subgraph', END)
      .compile();
    
    for await (const chunk of await graph.stream({ foo: 'foo [] ' })) {
      prettyPrint(chunk)
    }

    return {}
  }


  async callApi(prompt: string): Promise<ProviderResponse> {
    // return this.directCall(prompt);
    return this.indirectCall(prompt);
  }
    
}


export async function subgraphExample() {
  const example = new SubgraphExample();

  const prompt = '';

  const res = await example.callApi(prompt);

  prettyPrint(res);
}