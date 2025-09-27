import { ProviderResponse } from 'promptfoo';
import { Annotation, END, messagesStateReducer, START, StateGraph } from '@langchain/langgraph';

import { Example } from '../shared/base-example-api-provider';
import { prettyPrint } from '../utils';
import { BaseMessage, HumanMessage, SystemMessage } from '@langchain/core/messages';

export class LLMChainArch extends Example {

  private async llmChainGraph(userInput: string) {
    const stateAnnotation = Annotation.Root({
      messages: Annotation<BaseMessage[]>({
        reducer: messagesStateReducer,
        default: () => []
      }),
      userQuery: Annotation<string>(),
      sqlQuery: Annotation(),
      sqlExplain: Annotation(),
    });

    const generateSqlQueryNode = async (state: typeof stateAnnotation.State) => {
      const generateSqlPrompt = new SystemMessage('You are a helpful data analyst who generate SQL queries based on user questions.');
      const userMessage = new HumanMessage(state.userQuery);
      const messages = [generateSqlPrompt, ...state.messages, userMessage];

      this.model.temperature = 0.1;
      const res = await this.model.invoke(messages);

      return {
        sqlQuery: res.content as String,
        messages: [
          userMessage,
          res,
        ]
      }
    };

    const explainSqlQueryNode = async (state: typeof stateAnnotation.State) => {
      const explainSqlQueryPrompt = new SystemMessage("You are a helpful data analyst who explains SQL queries to users.");
      const messages = [explainSqlQueryPrompt, ...state.messages];

      this.model.temperature = 0.7;
      const res = await this.model.invoke(messages);

      return {
        sqlExplain: res.content as string,
        messages: [
          res
        ]
      }
    };

    const graph = new StateGraph(stateAnnotation)
      .addNode('generate_sql', generateSqlQueryNode)
      .addNode('explain_sql', explainSqlQueryNode)
      .addEdge(START, 'generate_sql')
      .addEdge('generate_sql', 'explain_sql')
      .addEdge('explain_sql', END)
      .compile();

    const res = await graph.invoke({
      userQuery: userInput
    });

    return res;
  }

  async callApi(userQuery: string): Promise<ProviderResponse> {
    return {
      output: await this.llmChainGraph(userQuery)
    };
  }
}

export async function llmChainArchExample() {
  const memSys = new LLMChainArch({});

  const userQuery = 'What is the total sales for each product?';
  const res = await memSys.callApi(userQuery);

  prettyPrint(res);
}