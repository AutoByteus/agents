import os
from autobyteus.agent.agent import Agent
from autobyteus.llm.rpa.claudechat_llm import ClaudeChatLLM
from autobyteus.tools.google_search_ui import GoogleSearch
from autobyteus.tools.webpage_screenshot_taker import WebPageScreenshotTaker
from autobyteus.tools.webpage_source_extractor import WebPageSourceExtractor
from autobyteus.prompt.prompt_builder import PromptBuilder
from autobyteus.llm.claude_models import ClaudeModel

claude_llm = ClaudeChatLLM(model=ClaudeModel.CLAUDE_3_OPUS)

async def main():
    role = "AI Movie Review Creator"
    current_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_file = os.path.join(current_dir, "movie_review_creator.prompt")
    
    prompt_builder = PromptBuilder.with_template(prompt_file)
    prompt = prompt_builder.variables(movie_topic="encouraging movie for students").build()

    llm = ClaudeChatLLM(ClaudeModel.CLAUDE_3_OPUS)
    tools = [
        GoogleSearch(),
        WebPageScreenshotTaker(),
        WebPageSourceExtractor()
    ]

    agent = Agent(role=role, prompt=prompt, llm=llm, tools=tools)
    await agent.run()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())