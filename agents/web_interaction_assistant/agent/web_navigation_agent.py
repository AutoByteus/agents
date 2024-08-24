import asyncio
import logging
from typing import List
from autobyteus.agent.agent import StandaloneAgent
from autobyteus.prompt.prompt_builder import PromptBuilder
from autobyteus.llm.base_llm import BaseLLM
from autobyteus.tools.base_tool import BaseTool
from autobyteus.tools.browser.session_aware.browser_session_aware_webpage_screenshot_taker import BrowserSessionAwareWebPageScreenshotTaker
from autobyteus.tools.browser.session_aware.browser_session_aware_web_element_trigger import BrowserSessionAwareWebElementTrigger

logger = logging.getLogger(__name__)

class WebNavigationAgent(StandaloneAgent):
    def __init__(self, role: str, prompt_builder: PromptBuilder, llm: BaseLLM, tools: List[BaseTool], **kwargs):
        super().__init__(role, prompt_builder, llm, tools, **kwargs)

    async def run(self):
        try:
            logger.info(f"Starting execution for WebNavigationAgent: {self.role}")
            self._initialize_task_completed()
            conversation_name = self._sanitize_conversation_name(self.role)
            self.conversation = await self.conversation_manager.start_conversation(
                conversation_name=conversation_name,
                llm=self.llm,
                persistence_provider_class=self.persistence_provider_class
            )
            logger.info(f"Conversation started for WebNavigationAgent: {self.role}")

            prompt = self.prompt_builder.set_variable_value("external_tools", self._get_external_tools_section()).build()
            logger.debug(f"Initial prompt for WebNavigationAgent {self.role}: {prompt}")

            response = await self.conversation.send_user_message(prompt)
            logger.info(f"Received initial LLM response for WebNavigationAgent {self.role}")

            while not self.task_completed.is_set():
                tool_invocation = self.response_parser.parse_response(response)

                if tool_invocation.is_valid():
                    name = tool_invocation.name
                    arguments = tool_invocation.arguments
                    logger.info(f"WebNavigationAgent {self.role} attempting to execute tool: {name}")

                    tool = next((t for t in self.tools if t.get_name() == name), None)
                    if tool:
                        try:
                            result = await tool.execute(**arguments)
                            logger.info(f"Tool '{name}' executed successfully by WebNavigationAgent {self.role}. Result: {result}")
                            
                            if self._requires_file_sending(tool):
                                response = await self.conversation.send_file(result)
                                logger.info(f"File sent to conversation for tool '{name}' by WebNavigationAgent {self.role}")
                            else:
                                response = await self.conversation.send_user_message(result)
                        except Exception as e:
                            error_message = str(e)
                            logger.error(f"Error executing tool '{name}' by WebNavigationAgent {self.role}: {error_message}")
                            response = await self.conversation.send_user_message(error_message)
                    else:
                        logger.warning(f"Tool '{name}' not found for WebNavigationAgent {self.role}.")
                        break
                else:
                    logger.info(f"Assistant response for WebNavigationAgent {self.role}: {response}")
                    await asyncio.sleep(1)
            
            logger.info(f"WebNavigationAgent {self.role} finished execution")
        finally:
            await self.cleanup()

    def _requires_file_sending(self, tool: BaseTool) -> bool:
        return isinstance(tool, (BrowserSessionAwareWebPageScreenshotTaker, BrowserSessionAwareWebElementTrigger))