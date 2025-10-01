import { toolCallingAgent } from './1-tool-calling-agent';
import { toolCallFirstAgent } from './2-tool-call-first-agent';
import { multiToolHandlingAgent } from './3-multi-tool-handling-agent';


(async () => {
  // await toolCallingAgent();
  // await toolCallFirstAgent();
  await multiToolHandlingAgent();
})();