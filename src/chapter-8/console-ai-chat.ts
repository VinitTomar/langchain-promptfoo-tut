import { SystemMessage, HumanMessage, AIMessageChunk } from '@langchain/core/messages';
import { ChatPromptTemplate } from '@langchain/core/prompts';
import { PostgresSaver } from "@langchain/langgraph-checkpoint-postgres";
import chalk from 'chalk';
import { ProviderResponse } from 'promptfoo';
import { read } from 'read';
import { v4 } from 'uuid';

import { Example } from '../shared/base-example-api-provider';
import { connectionUrl } from '../shared/pg-db-conn';
import { Annotation, CompiledStateGraph, END, Graph, MemorySaver, messagesStateReducer, START, StateGraph } from '@langchain/langgraph';
import { prettyPrint } from '../utils';
import { UserChatThreadRepo, UserRepo } from './pg-db-repo';


class AIAssistant extends Example {

  checkpointer = PostgresSaver.fromConnString(connectionUrl());

  stateAnnotation = Annotation.Root({
    messages: Annotation({
      reducer: messagesStateReducer,
      default: () => []
    }),
    userInput: Annotation<HumanMessage>(),
    latestAIReply: Annotation<string>(),
  });

  graph: CompiledStateGraph<
    typeof this.stateAnnotation.State,
    typeof this.stateAnnotation.Update,
    "__start__" | "llm",
    typeof this.stateAnnotation.spec
  >;

  private llmNode = async (state: typeof this.stateAnnotation.State) => {
    const msgs = [...state.messages, state.userInput];
    const res = await this.model.invoke(msgs);

    return {
      messages: res,
      latestAIReply: res.content.toString()
    }
  }

  constructor() {
    super();

    this.graph = new StateGraph(this.stateAnnotation)
      .addNode('llm', this.llmNode)
      .addEdge(START, 'llm')
      .addEdge('llm', END)
      .compile({ checkpointer: this.checkpointer });
  }

  private getThreadConfig(threadId: string) {
    return {
      configurable: { thread_id: threadId }
    };
  }

  async getThreadState(threadId: string) {
    this.stateAnnotation.State

    const state =  await this.graph.getState(this.getThreadConfig(threadId));

    return state.values;
  }

  async aiConversation(threadId: string, userInput: string): Promise<
    typeof this.stateAnnotation.State.latestAIReply
  > {
    const res = await this.graph.invoke({
      messages: new HumanMessage(userInput),
      userInput: new HumanMessage(userInput)
    }, this.getThreadConfig(threadId));

    return <string>res.latestAIReply;
  }

  async callApi(input: string): Promise<ProviderResponse> {

    const prompt = ChatPromptTemplate.fromMessages([
      /**
       * NOTES:
       * 
       * Using ['system', 'Instruction to model'] is more restrictive than new SystemMessage().
       * 
       * For example:
       * When this question is "Which year it is?" then model replies "I don't know the answer"
       * according to instructions, if we use ['system', 'Instruction to model'].
       * 
       * Else it gives year 2023, if we use "new SystemMessage()"
       */
      new SystemMessage(`You are a helpful assistant that answers to a human question. If you don't know the answer, then reply with "I don't know the answer to {question}. Please ask something else.`),
      new HumanMessage('User question: {question}')
    ]);

    const chatBot = prompt.pipe(this.model);
    
    const reply = await chatBot.invoke({ question: input})

    return {
      output: reply.content.toString()
    }
  }

}



class ConsoleChat {
  aiAssistant = new AIAssistant();
  formatToInstruction = chalk.cyan;
  printAIMsg = (msg: string) => console.log(chalk.blue(msg));
  printHumanMsg = (msg: string) => console.log(chalk.green(msg));

  userRepo = new UserRepo();
  userChatThreadRepo = new UserChatThreadRepo();

  private printInstruction(msg: string) {
    console.log(
      this.formatToInstruction(msg)
    )
  }

  private async printChatConversation(chatId: string) {
    const conv: typeof this.aiAssistant.stateAnnotation.State.messages = (await this.aiAssistant.getThreadState(chatId)).messages;

    if (!conv) {
      return;
    }

    for (const msg of conv) {
      if (msg.getType() === 'ai') {
        this.printAIMsg(
          '======================= AI =======================\n' +
          msg.content.toString()
        );
      } else {
        this.printHumanMsg(
          '======================= Human =======================\n' +
          msg.content.toString()
        );
      }
    }
  }

  private async chatting (chatId: string, title: string) {
    console.log(chalk.yellow(`========     ${title}     ========`));

    await this.printChatConversation(chatId);
    
    while (true) {
      const query = await read({
        prompt: 'Query> '
      });

      if (query.toLowerCase() === 'exit') {
        console.log(chalk.yellow('Chat is closed. Bye...'));
        break;
      }

      const reply = await this.aiAssistant.aiConversation(chatId, query);

      this.printAIMsg(reply.toString())
    }
  };

  private async startNewChat(username: string, prompt: string) {
    const history = await this.userChatThreadRepo.findByUsername(username);

    if (!history) {
      throw "History object not init";
    }

    const newId = v4();
    const title = await read({ prompt: this.formatToInstruction(prompt) });

    await this.userChatThreadRepo.createThread({
      id: newId,
      title,
      username,
    });

    await this.chatting(newId, title);
  }

  private async continueOldChat(username: string, index: string) {
    const history = await this.userChatThreadRepo.findByUsername(username);

    if (!history) {
      throw "History object not init";
    }

    const i = parseInt(index);
    if (Number.isNaN(i)) {
      throw `Invalid choice ${index}`;
    }
    const { id, title } = history[i - 1];
    await this.chatting(id, title);
  }
  
  private async initChat(username: string) {
    while (true) {
     const history = await this.userChatThreadRepo.findByUsername(username);

      if (history && history.length) {
        history.forEach((h, i) => {
          this.printInstruction(`${i + 1}) ${h.title}`)
        });

        const hIndex = await read({
          prompt: this.formatToInstruction(
            'Enter chat number to start old conversation or "new" for a new conversation.\n'
          )
        });

        if (hIndex === '-1') {
          console.log(chalk.blue(`Logging out ${username}. Bye...`));
          break;
        }

        if (hIndex === 'new') {
          await this.startNewChat(username, 'Enter new title: ');
        } else {
          await this.continueOldChat(username, hIndex);
        }
      } else {
        await this.startNewChat(username, 'Enter chat title to start chatting.\nTitle: ');
      } 
    }
  }

  private async login() {
    const username = await read({
      prompt:
        this.formatToInstruction('Enter username: ')
    });

    const password = await read({
      prompt: this.formatToInstruction('Enter password: '),
      silent: true,
      replace: '*',
    });

    const currentUser = await this.userRepo.findByUsername(username);

    if (currentUser?.password === password) {
      console.log(chalk.green(`User ${username} logged in successfully`));
      await this.initChat(username);
    } else {
      console.log(chalk.red(`Invalid username and password`));
    }
  }

  private async signup() {
    let username = await read({
      prompt:
        this.formatToInstruction('Enter new username: ')
    });


    while (!!(await this.userRepo.findByUsername(username))) {
      console.log(chalk.red(`User ${username} exist. Please enter another username.`));
        username = await read({
        prompt:
          this.formatToInstruction('Enter new username: ')
        });
    }

    const password = await read({
      prompt: this.formatToInstruction('Enter password: '),
      silent: true,
      replace: '*',
    });

    this.userRepo.create({ username, password });
    this.printInstruction(`User ${username} is registered successfully.`);
    await this.initChat(username);
  }

  async start() {
    console.log(this.formatToInstruction(`Welcome to Console AI chat bot.`));
    while (true) {
      const input = await read({
        prompt:
          this.formatToInstruction('1)Login, 2)Signup => ')
      });

      switch (input) {
        case '1':
          await this.login();
          break;
        
        case '2':
          await this.signup();
          break;

        case 'end':
          return;
      }
    }
  }
}

export async function consoleAIChatExample() {
  new ConsoleChat().start();
}