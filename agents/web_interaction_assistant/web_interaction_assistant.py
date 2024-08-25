import logging
import os
from autobyteus.agent.agent import StandaloneAgent
from autobyteus.llm.rpa.claudechat_llm import ClaudeChatLLM
from autobyteus.llm.rpa.gemini_llm import GeminiLLM
from autobyteus.llm.rpa.chatgpt_llm import ChatGPTLLM
from autobyteus.tools.browser.standalone.google_search_ui import GoogleSearch
from autobyteus.tools.ask_user_input import AskUserInput
from autobyteus.tools.browser.standalone.webpage_screenshot_taker import WebPageScreenshotTaker
from autobyteus.tools.browser.standalone.webpage_reader import WebPageReader
from autobyteus.tools.browser.session_aware.browser_session_aware_webpage_reader import BrowserSessionAwareWebPageReader
from autobyteus.tools.browser.session_aware.browser_session_aware_web_element_trigger import BrowserSessionAwareWebElementTrigger
from autobyteus.tools.browser.session_aware.browser_session_aware_webpage_screenshot_taker import BrowserSessionAwareWebPageScreenshotTaker
from autobyteus.tools.image_downloader import ImageDownloader
from autobyteus_community_tools.social_media_poster.weibo.weibo_poster import WeiboPoster
from autobyteus_community_tools.social_media_poster.weibo.reviewed_movies_retriever import ReviewedMoviesRetriever
from autobyteus.prompt.prompt_builder import PromptBuilder
from autobyteus.llm.models import LLMModel

from agent.web_navigation_agent import WebNavigationAgent


def setup_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create file handler
    file_handler = logging.FileHandler('book_review_creator.log')
    file_handler.setLevel(logging.INFO)

    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

async def main():
    # Set up logger
    setup_logger()
    logger = logging.getLogger(__name__)

    logger.info("Starting General Web Interaction Assistant")

    role = "General Web Interaction Assistant"
    current_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_file = os.path.join(current_dir, "web_interaction_assistant_new.prompt")
    
    prompt_builder = PromptBuilder.from_file(prompt_file).set_variable_value(name="user_task", value="visit the https://pubmed.ncbi.nlm.nih.gov/, search paper based on what user want to search for, and pick one paper, find out how to download it, and finally download the pdf.")

    llm = ClaudeChatLLM(LLMModel.CLAUDE_3_5_SONNET)
    #llm = GeminiLLM()
    #llm = ChatGPTLLM(model="Default")
    tools = [
        BrowserSessionAwareWebPageReader(),
        BrowserSessionAwareWebPageScreenshotTaker(),
        BrowserSessionAwareWebElementTrigger(),
        AskUserInput(),
    ]

    agent = WebNavigationAgent(role=role, prompt_builder= prompt_builder, llm=llm, tools=tools)
    await agent.run()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
