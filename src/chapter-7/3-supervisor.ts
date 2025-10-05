import { ProviderResponse } from 'promptfoo';
import { Annotation, END, MessagesAnnotation, messagesStateReducer, START, StateGraph } from '@langchain/langgraph';
import { AIMessage, HumanMessage, SystemMessage } from '@langchain/core/messages';
import { z } from 'zod';

import { Example } from '../shared/base-example-api-provider';
import { prettyPrint } from '../utils';



export class SupervisorExample extends Example {
  async callApi(prompt: string): Promise<ProviderResponse> {

    const stateAnnotation = Annotation.Root({
      ...MessagesAnnotation.spec,
      next: Annotation<string>(),
    })

    const agents = ['researcher', 'coder'];

    const supervisorDecision = z.object({
      next: z.enum(['FINISH', ...agents])
    });

    const supervisorNode = async (state: typeof stateAnnotation.State) => {
      /**
       * TODO
       * Need to find a better prompt. It finishes after two iteration to the same agent.
       */

      const sysPrompt1 = new SystemMessage(`
        You are a supervisor tasked with managing a conversation between the following workers: ${agents.join(', ')}. Given the following user request, respond with the worker to act next. Each worker will perform a task and respond with their results and status.

        Check the whole conversation if user's query is already answered respond with FINISH. Do not return to worker.
      `);

      /**
       * Need to remove second system prompt. Getting error
       * Error("System message should be the first one")
       */

      // const sysPrompt2 = new SystemMessage(`
          //  Given the conversation above, who should act next? Or should we FINISH? Select one of: ${agents.join(', ')}, FINISH
      // `);

      this.model.temperature = 0;
      const model = this.model.withStructuredOutput(supervisorDecision);

      return await model.invoke([
        sysPrompt1,
        // sysPrompt2,
        ...state.messages,
      ]);
    };

    const researcherNode = async (state: typeof stateAnnotation.State) => {
      const researchPrompt = new SystemMessage(`
          You are a researcher whose job is to find the result based on user's query.
        `);
      /**
       * In real world example we should use different kind of model based on usecase.
       */
      const res = await this.model.invoke([
        researchPrompt,
        state.messages[0]
      ]);

      return {
        messages: res
      }
    };

    const coderNode = async (state: typeof stateAnnotation.State) => {
      const coderPrompt = new SystemMessage(`
          You are a coder whose job is to generate code for user's query.
        `);
      /**
       * In real world example we should use different kind of model based on usecase.
       */
      const res = await this.model.invoke([
        coderPrompt,
        state.messages[0]
      ]);

      return {
        messages: res
      }
    };

    const conditionalEdge = async (state: typeof stateAnnotation.State) => {
      return state.next === 'FINISH' ? END : state.next;
    }

    const graph = new StateGraph(stateAnnotation)
      .addNode('supervisor', supervisorNode)
      .addNode(agents[0], researcherNode)
      .addNode(agents[1], coderNode)
      .addEdge(START, 'supervisor')
      .addEdge('researcher', 'supervisor')
      .addEdge('coder', 'supervisor')
      .addConditionalEdges('supervisor', conditionalEdge)
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


export async function supervisorExample() {
  const example = new SupervisorExample();

  const prompt = 'Write a python code for Hello world';

  const res = await example.callApi(prompt);

  prettyPrint(res);
}