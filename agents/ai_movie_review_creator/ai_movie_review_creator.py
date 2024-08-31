import os
import logging
import asyncio
from autobyteus.agent.orchestrator.single_replica_agent_orchestrator import SingleReplicaAgentOrchestrator
from autobyteus.agent.group.group_aware_agent import GroupAwareAgent
from autobyteus.agent.group.coordinator_agent import CoordinatorAgent
from autobyteus.llm.rpa.gemini_llm import GeminiLLM
from autobyteus.tools.browser.standalone.google_search_ui import GoogleSearch
from autobyteus.tools.browser.standalone.webpage_reader import WebPageReader, CleaningMode
from autobyteus.tools.image_downloader import ImageDownloader
from autobyteus_community_tools.social_media_poster.weibo.weibo_poster import WeiboPoster
from autobyteus_community_tools.social_media_poster.weibo.reviewed_movies_retriever import ReviewedMoviesRetriever
from autobyteus.prompt.prompt_builder import PromptBuilder
from autobyteus.llm.models import LLMModel
from autobyteus.llm.rpa.claudechat_llm import ClaudeChatLLM
from autobyteus.llm.rpa.perplexity_llm import PerplexityLLM
from autobyteus.llm.rpa.groq_llm import GroqLLM
from autobyteus.llm.rpa.mistral_llm import MistralLLM

def setup_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler('ai_movie_review_creator.log')

    console_handler.setLevel(logging.INFO)
    file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

def setup_agent_group():
    singleReplicaAgentOrchestrator = SingleReplicaAgentOrchestrator()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    prompts_dir = os.path.join(current_dir, "prompts")

    # GoogleSearchAgent
    google_search_prompt = os.path.join(prompts_dir, "GoogleSearchAgent.prompt")
    google_search_llm = PerplexityLLM(model=LLMModel.LLAMA_3_1_70B_INSTRUCT)#MistralLLM(model=LLMModel.MISTRAL_LARGE)  #GeminiLLM(model=LLMModel.GEMINI_1_5_PRO_EXPERIMENTAL)
    google_search_prompt = PromptBuilder().from_file(google_search_prompt)
    google_search_tools = [GoogleSearch(cleaning_mode=CleaningMode.GOOGLE_SEARCH_RESULT)]
    google_search_agent = GroupAwareAgent("GoogleSearchAgent", google_search_prompt, google_search_llm, google_search_tools)

    # WebPageReaderAgent
    webpage_reader_prompt = os.path.join(prompts_dir, "WebPageReaderAgent.prompt")
    webpage_reader_llm = GeminiLLM(model=LLMModel.GEMINI_1_5_PRO_EXPERIMENTAL)
    webpage_reader_prompt = PromptBuilder().from_file(webpage_reader_prompt)
    webpage_reader_tools = [WebPageReader(cleaning_mode=CleaningMode.TEXT_CONTENT_FOCUSED)]
    webpage_reader_agent = GroupAwareAgent("WebPageReaderAgent", webpage_reader_prompt, webpage_reader_llm, webpage_reader_tools)

    # ReviewWritingAgent
    review_writing_prompt = os.path.join(prompts_dir, "ReviewWritingAgent.prompt")
    review_writing_llm = ClaudeChatLLM(model=LLMModel.CLAUDE_3_5_SONNET)  #GeminiLLM(model=LLMModel.GEMINI_1_5_PRO_EXPERIMENTAL)
    review_writing_prompt = PromptBuilder().from_file(review_writing_prompt)
    review_writing_tools = []
    review_writing_agent = GroupAwareAgent("ReviewWritingAgent", review_writing_prompt, review_writing_llm, review_writing_tools)

    # WeiboPosterAgent
    weibo_poster_prompt = os.path.join(prompts_dir, "WeiboPosterAgent.prompt")
    weibo_poster_llm = GeminiLLM(model=LLMModel.GEMINI_1_5_PRO_EXPERIMENTAL)
    weibo_poster_prompt = PromptBuilder().from_file(weibo_poster_prompt)
    weibo_poster_tools = [WeiboPoster(weibo_account_name="RyanZhengHaliluya")]
    weibo_poster_agent = GroupAwareAgent("WeiboPosterAgent", weibo_poster_prompt, weibo_poster_llm, weibo_poster_tools)

    # Add agents to the group
    singleReplicaAgentOrchestrator.add_agent(google_search_agent)
    singleReplicaAgentOrchestrator.add_agent(webpage_reader_agent)
    singleReplicaAgentOrchestrator.add_agent(review_writing_agent)
    singleReplicaAgentOrchestrator.add_agent(weibo_poster_agent)

    # CoordinatorAgent
    coordinator_prompt = os.path.join(prompts_dir, "CoordinationAgent.prompt")
    coordinator_llm = GeminiLLM(model=LLMModel.GEMINI_1_5_PRO_EXPERIMENTAL)
    coordinator_prompt = PromptBuilder().from_file(coordinator_prompt).set_variable_value(name="movie_topic", value="encourge movies for young mums")
    coordinator_tools = [ReviewedMoviesRetriever()]
    coordinator_agent = CoordinatorAgent("CoordinationAgent", coordinator_prompt, coordinator_llm, coordinator_tools)
    singleReplicaAgentOrchestrator.set_coordinator_agent(coordinator_agent)

    return singleReplicaAgentOrchestrator

async def run_movie_review_workflow(agent_group: SingleReplicaAgentOrchestrator):
    result = await agent_group.run()
    print(result)

async def main():
    setup_logger()
    logger = logging.getLogger(__name__)

    logger.info("Starting AI Movie Review Creator")

    agent_group = setup_agent_group()
    await run_movie_review_workflow(agent_group)

    logger.info("AI Movie Review Creator completed")

if __name__ == "__main__":
    asyncio.run(main())

