import os
from autobyteus.agent.agent import Agent
from autobyteus.llm.rpa.claudechat_llm import ClaudeChatLLM
from autobyteus.llm.rpa.gemini_llm import GeminiLLM
from autobyteus.llm.rpa.groq_llm import GroqLLM
from autobyteus.llm.rpa.chatgpt_llm import ChatGPTLLM
from autobyteus.llm.rpa.mistral_llm import MistralLLM
from autobyteus.tools.browser.standalone.google_search_ui import GoogleSearch
from autobyteus.tools.browser.standalone.webpage_screenshot_taker import WebPageScreenshotTaker
from autobyteus.tools.browser.standalone.webpage_reader import WebPageReader, CleaningMode
from autobyteus.tools.image_downloader import ImageDownloader
from autobyteus.tools.social_media_poster.weibo.weibo_poster import WeiboPoster
from autobyteus.tools.social_media_poster.weibo.reviewed_movies_retriever import ReviewedMoviesRetriever
from autobyteus.prompt.prompt_builder import PromptBuilder
from autobyteus.llm.models import LLMModel


async def main():
    role = "AI Movie Review Creator"
    current_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_file = os.path.join(current_dir, "simplified.prompt")
    
    prompt_builder = PromptBuilder.from_template(prompt_file).set_variable_value(name="movie_topic", value="encouraging movies for education")

    llm = ClaudeChatLLM(LLMModel.CLAUDE_3_5_SONNET)
    #llm = GeminiLLM()
    #llm = ChatGPTLLM(model="Default")
    #llm = GroqLLM(model=GroqModel.LLAMA_3_1_70B_VERSATILE)
    #llm = MistralLLM(model = MistralModel.MISTRAL_LARGE)
    tools = [
        GoogleSearch(cleaning_mode=CleaningMode.ULTIMATE),
        ReviewedMoviesRetriever(),
        WebPageReader(content_cleanup_level=CleaningMode.ULTIMATE),
        ImageDownloader(),
        WeiboPoster(weibo_account_name="Normy-光影旅程")
    ]

    agent = Agent(role=role, prompt_builder= prompt_builder, llm=llm, tools=tools)
    await agent.run()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
