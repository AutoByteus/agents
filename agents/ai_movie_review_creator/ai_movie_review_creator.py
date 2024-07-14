import os
from autobyteus.agent.agent import Agent
from autobyteus.llm.rpa.claudechat_llm import ClaudeChatLLM
from autobyteus.llm.rpa.gemini_llm import GeminiLLM
from autobyteus.llm.rpa.chatgpt_llm import ChatGPTLLM
from autobyteus.tools.google_search_ui import GoogleSearch
from autobyteus.tools.webpage_screenshot_taker import WebPageScreenshotTaker
from autobyteus.tools.webpage_reader import WebPageReader
from autobyteus.tools.image_downloader import ImageDownloader
from autobyteus.tools.weibo.weibo_poster import WeiboPoster
from autobyteus.prompt.prompt_builder import PromptBuilder
from autobyteus.llm.claude_models import ClaudeModel


async def main():
    role = "AI Movie Review Creator"
    current_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_file = os.path.join(current_dir, "simpler_xml_v2.prompt")
    
    prompt_builder = PromptBuilder.from_template(prompt_file).set_variable_value(name="movie_topic", value="encouraging movie for students")

    llm = ClaudeChatLLM(ClaudeModel.CLAUDE_3_5_SONNET)
    #llm = GeminiLLM()
    #llm = ChatGPTLLM(model="Default")
    tools = [
        GoogleSearch(),
        WebPageReader(),
        ImageDownloader(),
        WeiboPoster(weibo_account_name="humphreyZheng")
    ]

    agent = Agent(role=role, prompt_builder= prompt_builder, llm=llm, tools=tools)
    await agent.run()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
