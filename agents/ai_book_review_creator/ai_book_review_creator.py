import os
import logging
import asyncio
from autobyteus.agent.orchestrator.single_replica_agent_orchestrator import SingleReplicaAgentOrchestrator
from autobyteus.agent.group.group_aware_agent import GroupAwareAgent
from autobyteus.agent.group.coordinator_agent import CoordinatorAgent
from autobyteus.llm.rpa.claudechat_llm import ClaudeChatLLM
from autobyteus.llm.rpa.gemini_llm import GeminiLLM
from autobyteus.llm.rpa.groq_llm import GroqLLM
from autobyteus.llm.rpa.mistral_llm import MistralLLM
from autobyteus.tools.browser.standalone.google_search_ui import GoogleSearch
from autobyteus.tools.browser.standalone.webpage_reader import WebPageReader, CleaningMode
from autobyteus.tools.image_downloader import ImageDownloader
from autobyteus_community_tools.social_media_poster.xiaohongshu.xiaohongshu_poster import XiaohongshuPoster
from autobyteus_community_tools.social_media_poster.xiaohongshu.reviewed_books_retriever import ReviewedBooksRetriever
from autobyteus.prompt.prompt_builder import PromptBuilder
from autobyteus.llm.models import LLMModel

def setup_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler('book_review_creator.log')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    for handler in [console_handler, file_handler]:
        handler.setFormatter(formatter)
        logger.addHandler(handler)

def setup_agent_group():
    singleReplicaAgentOrchestrator = SingleReplicaAgentOrchestrator()
    current_dir = "/Users/qingwang/learning/agents/agents/ai_book_review_creator/prompts"
    
    # Common tools
    google_search = GoogleSearch(cleaning_mode=CleaningMode.GOOGLE_SEARCH_RESULT)
    webpage_reader = WebPageReader(cleaning_mode=CleaningMode.TEXT_CONTENT_FOCUSED)
    
    # Topic Understanding Agent
    topic_understanding_prompt = PromptBuilder().from_file(os.path.join(current_dir, "GoogleSearchAgent.prompt"))
    topic_understanding_agent = GroupAwareAgent("GoogleSearchAgent", topic_understanding_prompt, GeminiLLM(model=LLMModel.GEMINI_1_5_PRO), [google_search, webpage_reader])
    
    # Book Selection Agent
    book_selection_prompt = PromptBuilder().from_file(os.path.join(current_dir, "BookSelectionAgent.prompt"))
    book_selection_agent = GroupAwareAgent("BookSelectionAgent", book_selection_prompt,GeminiLLM(model=LLMModel.GEMINI_1_5_PRO), [google_search, ReviewedBooksRetriever()])
    
    # Information Gathering Agent
    info_gathering_prompt = PromptBuilder().from_file(os.path.join(current_dir, "InformationGatheringAgent.prompt"))
    info_gathering_agent = GroupAwareAgent("InformationGatheringAgent", info_gathering_prompt, GeminiLLM(model=LLMModel.GEMINI_1_5_PRO), [google_search, webpage_reader])
    
    # Review Writing Agent
    review_writing_prompt = PromptBuilder().from_file(os.path.join(current_dir, "ReviewWritingAgent.prompt"))
    review_writing_agent = GroupAwareAgent("ReviewWritingAgent", review_writing_prompt, GeminiLLM(model=LLMModel.GEMINI_1_5_PRO_EXPERIMENTAL), [])
    
    # Image Acquisition Agent
    #image_acquisition_prompt = PromptBuilder().from_file(os.path.join(current_dir, "ImageAcquisitionAgent.prompt"))
    #image_acquisition_agent = GroupAwareAgent("ImageAcquisitionAgent", image_acquisition_prompt, MistralLLM(model=LLMModel.MISTRAL_LARGE), [ImageDownloader()])
    
    # Publishing Agent
    publishing_prompt = PromptBuilder().from_file(os.path.join(current_dir, "PublishingAgent.prompt"))
    publishing_agent = GroupAwareAgent("PublishingAgent", publishing_prompt, ClaudeChatLLM(model=LLMModel.CLAUDE_3_5_SONNET), [XiaohongshuPoster(xiaohongshu_account_name="Normy-光影旅程")])
    
    # Add agents to the orchestrator
    for agent in [topic_understanding_agent, book_selection_agent, info_gathering_agent, review_writing_agent , publishing_agent]:
        singleReplicaAgentOrchestrator.add_agent(agent)
    
    # Set up Coordinator Agent
    coordinator_prompt = PromptBuilder().from_file(os.path.join(current_dir, "CoordinationAgent.prompt"))
    coordinator_agent = CoordinatorAgent("CoordinationAgent", coordinator_prompt, GeminiLLM(model=LLMModel.GEMINI_1_5_PRO_EXPERIMENTAL), [])
    singleReplicaAgentOrchestrator.set_coordinator_agent(coordinator_agent)
    
    return singleReplicaAgentOrchestrator

async def run_book_review_workflow(agent_group: SingleReplicaAgentOrchestrator):
    result = await agent_group.run()
    logging.info("Book Review Creator Workflow Result:\n%s", result)

async def main():
    setup_logger()
    logging.info("Starting Book Review Creator")
    agent_group = setup_agent_group()
    await run_book_review_workflow(agent_group)
    logging.info("Book Review Creator completed")

if __name__ == "__main__":
    asyncio.run(main())