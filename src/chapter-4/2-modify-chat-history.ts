import { ProviderResponse } from 'promptfoo';

import { Example } from '../shared/base-example-api-provider';
import { prettyPrint } from '../utils';
import { trimMessages, SystemMessage, HumanMessage, AIMessage, filterMessages, BaseMessage, mergeMessageRuns } from '@langchain/core/messages';

export class ModifyChatHistory extends Example {

  private msgToJSON(msg: BaseMessage) {
    return {
      type: msg.getType(),
      content: msg.content,
    }
  }

  private async trimmingMsg() {
    const trimmer = trimMessages({
      maxTokens: 30,
      strategy: "last",
      tokenCounter: this.model,
      includeSystem: true,
      allowPartial: false,
      startOn: "human",
    });

    const messages = [
      new SystemMessage("you're a good assistant"),
      new HumanMessage("hi! I'm bob"),
      new AIMessage("hi!"),
      new HumanMessage("I like vanilla ice cream"),
      new AIMessage("nice"),
      new HumanMessage("what's 2 + 2"),
      new AIMessage("4"),
      new HumanMessage("thanks"),
      new AIMessage("no problem!"),
      new HumanMessage("having fun?"),
      new AIMessage("yes!"),
    ]

    const trimmed = await trimmer.invoke(messages);

    return trimmed.map(this.msgToJSON);
  }

  private async filterMessages() {
    const messages = [
      new SystemMessage({content: "you are a good assistant", id: "1"}),
      new HumanMessage({content: "example input", id: "2", name: "example_user"}),
      new AIMessage({content: "example output", id: "3", name: "example_assistant"}),
      new HumanMessage({content: "real input", id: "4", name: "bob"}),
      new AIMessage({content: "real output", id: "5", name: "alice"}),
    ];

    const msg = filterMessages(messages, {
      includeTypes: ["human", 'ai'],
      excludeIds: ['3']
    });

    return msg.map(this.msgToJSON);
  }

  private async mergingConsecutiveMessages() {
    const messages = [
      new SystemMessage("you're a good assistant."),
      new SystemMessage("you always respond with a joke."),
      new HumanMessage({
        content: [{ type: "text", text: "i wonder why it's called langchain" }],
      }),
      new HumanMessage("and who is harrison chasing anyway"),
      new AIMessage(
        `Well, I guess they thought "WordRope" and "SentenceString" just didn\'t 
          have the same ring to it!`
      ),
      new AIMessage(
        "Why, he's probably chasing after the last cup of coffee in the office!"
      ),
    ];

    const mergedMsg = mergeMessageRuns(messages);
    return mergedMsg.map(this.msgToJSON);
  }

  async callApi(prompt: string): Promise<ProviderResponse> {
    return {
      output: {
        // trimmed: await this.trimmingMsg()
        // filtered: await this.filterMessages()
        merged: await this.mergingConsecutiveMessages()
      }
    }
  }
  
}

export async function modifyChatHistory() {
  const ins = new ModifyChatHistory({});

  const query = '';

  const res = await ins.callApi(query);

  prettyPrint(res);
}