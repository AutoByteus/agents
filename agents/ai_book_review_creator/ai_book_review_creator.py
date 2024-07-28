# File: autobyteus/scripts/book_review_creator.py

import os
import logging
import asyncio
from autobyteus.agent.agent import Agent
from autobyteus.llm.rpa.claudechat_llm import ClaudeChatLLM
from autobyteus.llm.rpa.gemini_llm import GeminiLLM
from autobyteus.llm.rpa.groq_llm import GroqLLM, GroqModel
from autobyteus.llm.rpa.chatgpt_llm import ChatGPTLLM
from autobyteus.llm.rpa.mistral_llm import MistralLLM, MistralModel
from autobyteus.tools.browser.standalone.google_search_ui import GoogleSearch
from autobyteus.tools.browser.standalone.webpage_screenshot_taker import WebPageScreenshotTaker
from autobyteus.tools.browser.standalone.webpage_reader import WebPageReader, CleaningMode
from autobyteus.tools.image_downloader import ImageDownloader
from autobyteus.tools.social_media_poster.weibo.weibo_poster import WeiboPoster
from autobyteus.tools.social_media_poster.xiaohongshu.xiaohongshu_poster import XiaohongshuPoster
from autobyteus.tools.social_media_poster.xiaohongshu.reviewed_books_retriever import ReviewedBooksRetriever
from autobyteus.tools.social_media_poster.weibo.reviewed_movies_retriever import ReviewedMoviesRetriever
from autobyteus.prompt.prompt_builder import PromptBuilder
from autobyteus.llm.claude_models import ClaudeModel

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

    logger.info("Starting Book Review Creator")

    role = "AI Book Review Creator"
    current_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_file = os.path.join(current_dir, "book_review_creator_v2.prompt")
    
    prompt_builder = PromptBuilder.from_template(prompt_file).set_variable_value(name="book_topic", value="encouraging book for education")

    llm = ClaudeChatLLM(ClaudeModel.CLAUDE_3_5_SONNET)
    #llm = GeminiLLM()
    #llm = ChatGPTLLM(model="Default")
    #llm = GroqLLM(model=GroqModel.LLAMA_3_1_70B_VERSATILE)
    #llm = MistralLLM(model = MistralModel.MISTRAL_LARGE)
    tools = [
        GoogleSearch(cleaning_mode=CleaningMode.ULTIMATE),
        ReviewedBooksRetriever(),
        WebPageReader(content_cleanup_level=CleaningMode.ULTIMATE),
        ImageDownloader(),
        XiaohongshuPoster(xiaohongshu_account_name="Normy-光影旅程")
    ]

    logger.info("Initializing Agent")
    agent = Agent(role=role, prompt_builder=prompt_builder, llm=llm, tools=tools)
    
    logger.info("Running Agent")
    await agent.run()

    logger.info("Book Review Creator completed")

if __name__ == "__main__":
    asyncio.run(main())