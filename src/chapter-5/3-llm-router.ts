import { ProviderResponse } from 'promptfoo';
import { Annotation, END, messagesStateReducer, START, StateGraph } from '@langchain/langgraph';

import { ExampleWithEmbeddings } from '../shared/base-example-api-provider';
import { prettyPrint } from '../utils';
import { BaseMessage, HumanMessage, SystemMessage } from '@langchain/core/messages';
import { MemoryVectorStore } from 'langchain/vectorstores/memory';

export class LLMRouterArch extends ExampleWithEmbeddings {

  private async llmRouterGraph(userInput: string) {
    const stateAnnotation = Annotation.Root({
      messages: Annotation<BaseMessage[]>({
        reducer: messagesStateReducer,
        default: () => []
      }),
      domain: Annotation< 'records' | 'insurance' >(),
      userQuery: Annotation<string>(),
      documents: Annotation<[]>(),
      answer: Annotation(),
    });

    const retrieveMedicalRecordNode = async (state: typeof stateAnnotation.State) => {
      const medicalRecordStore = await MemoryVectorStore.fromDocuments([], this.embeddings);
      const medicalRecordRetriever = medicalRecordStore.asRetriever();

      const docs = await medicalRecordRetriever.invoke(state.userQuery);

      return {
        documents: docs
      }
    };

    const retrieveInsuranceFaqsNode = async (state: typeof stateAnnotation.State) => {
      const insuranceFaqsStore = await MemoryVectorStore.fromDocuments([], this.embeddings);
      const insuranceFaqsRetriever = insuranceFaqsStore.asRetriever();

      const docs = await insuranceFaqsRetriever.invoke(state.userQuery);

      return {
        documents: docs
      }
    };

    const routerNode = async (state: typeof stateAnnotation.State) => {
      const routerPrompt = new SystemMessage(`
        You need to decide which domain to route the user query to. You have two  domains to choose from:
          - records: contains medical records of the patient, such as diagnosis, treatment, and prescriptions.
          - insurance: contains frequently asked questions about insurance policies, claims, and coverage.

        Output only the domain name.
        `);
      const userMsg = new HumanMessage(state.userQuery);
      const messages = [routerPrompt, ...state.messages, userMsg];

      this.model.temperature = 0;
      const res = await this.model.invoke(messages);

      return {
        domain: res.content as 'records' | 'insurance',
        messages: [userMsg, res]
      }
    }

    const pickRouterEdge = (state: typeof stateAnnotation.State) => {
      if (state.domain === "records") {
        return "retrieve_medical_records";
      } else {
        return "retrieve_insurance_faqs";
      }
    }

    const generateAnswerNode = async (state: typeof stateAnnotation.State) => {
      const medicalRecordsPrompt = new SystemMessage(
        `You are a helpful medical chatbot who answers questions based on the 
          patient's medical records, such as diagnosis, treatment, and 
          prescriptions.`
      );

      const insuranceFaqsPrompt = new SystemMessage(
        `You are a helpful medical insurance chatbot who answers frequently asked 
          questions about insurance policies, claims, and coverage.`
      );

      const prompt = state.domain === "records" ? medicalRecordsPrompt : insuranceFaqsPrompt;
      const messages = [
        prompt,
        ...state.messages,
        new HumanMessage(`Documents: ${state.documents}`),
      ];

      this.model.temperature = 0.7;
      const res = await this.model.invoke(messages);

      return {
        answer: res.content,
        messages: res,
      }
    }

    const graph = new StateGraph(stateAnnotation)
      .addNode("router", routerNode)
      .addNode("retrieve_medical_records", retrieveMedicalRecordNode)
      .addNode("retrieve_insurance_faqs", retrieveInsuranceFaqsNode)
      .addNode("generate_answer", generateAnswerNode)
      .addEdge(START, "router")
      .addConditionalEdges("router", pickRouterEdge)
      .addEdge("retrieve_medical_records", "generate_answer")
      .addEdge("retrieve_insurance_faqs", "generate_answer")
      .addEdge("generate_answer", END)
      .compile();
    
    const input = {
      userQuery: userInput
    };
    
    for await (const chunk of await graph.stream(input)) {
      prettyPrint(chunk);
    }
  }

  async callApi(userQuery: string): Promise<ProviderResponse> {
    return {
      output: await this.llmRouterGraph(userQuery)
    };

  }
}

export async function llmRouterArchExample() {
  const llmRtr = new LLMRouterArch({});

  llmRtr.callApi("Am I covered for COVID-19 treatment?");

  // const userQuery = 'What is the total sales for each product?';
  // const res = await memSys.callApi(userQuery);

  // prettyPrint(res);
}