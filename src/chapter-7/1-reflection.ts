import { ProviderResponse } from 'promptfoo';
import { Annotation, END, messagesStateReducer, START, StateGraph } from '@langchain/langgraph';
import { AIMessage, HumanMessage, SystemMessage } from '@langchain/core/messages';

import { Example } from '../shared/base-example-api-provider';
import { prettyPrint } from '../utils';



export class ReflectionExample extends Example {
  async callApi(prompt: string): Promise<ProviderResponse> {
    const stateAnnotation = Annotation.Root({
      messages: Annotation({
        reducer: messagesStateReducer,
        default: () => []
      })
    });

    const generateNode = async (state: typeof stateAnnotation.State) => {
      const messages = [
        new SystemMessage(`
          You are an essay assistant tasked with writing excellent 3-paragraph essays. Generate the best essay possible for the user's request. If the user provides critique, respond with a revised version of your previous attempts.
        `),
        ...state.messages
      ];

      const res = await this.model.invoke(messages);

      return {
        messages: res
      }
    };
    
    const reviewerNode = async (state: typeof stateAnnotation.State) => {
      const reviewPrompt = new SystemMessage(`
        You are a teacher grading an essay submission. Generate critique and recommendations for the user's submission. Provide detailed recommendations, including requests for length, depth, style, etc.  
      `);

      const clsMap = {
        ai: HumanMessage,
        human: AIMessage,
      }

      const translatedMessages = [
        reviewPrompt,
        // first msg is the user input
        state.messages[0],
        ...state.messages.slice(1)
          // .filter(msg => Object.keys(clsMap).includes(msg.getType()))
          .map(msg => {
            return new clsMap[msg.getType() as 'ai' | 'human'](msg.content.toString())
          })
      ];

      const answer = await this.model.invoke(translatedMessages);
      
      return {
        messages: new HumanMessage(answer.content.toString())
      }
    };

    const shouldContinueEdge = (state: typeof stateAnnotation.State) => {
      if (state.messages.length > 6)
        return END;

      return 'review';
    }

    const graph = new StateGraph(stateAnnotation)
      .addNode('generate', generateNode)
      .addNode('review', reviewerNode)
      .addEdge(START, 'generate')
      .addEdge('review', 'generate')
      .addConditionalEdges('generate', shouldContinueEdge)
      .compile();
    
    const input = {
      messages: new HumanMessage(prompt)
    };

    for await (const chunk of await graph.stream(input)) {
      prettyPrint(chunk);
    }

    return {}
  }
}


export async function reflectionExample() {
  const example = new ReflectionExample();

  const prompt = 'Write an essay on Diwali';

  const res = await example.callApi(prompt);

  prettyPrint(res);
}