import asyncio
import os
import logging
from autobyteus.agent.group.single_replica_agent_orchestrator import SingleReplicaAgentOrchestrator
from autobyteus.agent.group.group_aware_agent import GroupAwareAgent
from autobyteus.agent.group.coordinator_agent import CoordinatorAgent
from autobyteus.llm.models import LLMModel
from autobyteus.llm.rpa.gemini_llm import GeminiLLM
from autobyteus.llm.rpa.claudechat_llm import ClaudeChatLLM
from autobyteus.llm.rpa.perplexity_llm import PerplexityLLM
from autobyteus.llm.rpa.groq_llm import GroqLLM
from autobyteus.llm.rpa.mistral_llm import MistralLLM
from autobyteus.prompt.prompt_builder import PromptBuilder
from autobyteus.tools.browser.standalone.webpage_reader import WebPageReader

def setup_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler('agent_workflow.log')

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

    # PubMedSearchAgent
    pubmed_search_prompt = os.path.join(current_dir, "PubMedSearchAgent.prompt")
    pubmed_search_llm = PerplexityLLM(LLMModel.LLAMA_3_1_SONAR_LARGE_128K_CHAT)
    pubmed_search_prompt = PromptBuilder().from_file(pubmed_search_prompt)
    pubmed_search_tools = [PubMedSearch()]
    pubmed_search_agent = GroupAwareAgent("PubMedSearchAgent", pubmed_search_prompt, pubmed_search_llm, pubmed_search_tools)

    # AdaptiveDownloadFinderAgent
    download_finder_prompt = os.path.join(current_dir, "AdaptiveDownloadFinderAgent.prompt")
    download_finder_llm = GroqLLM(model=LLMModel.LLAMA_3_1_70B_VERSATILE)
    download_finder_prompt = PromptBuilder().from_file(download_finder_prompt)
    download_finder_tools = [WebPageReader()]
    download_finder_agent = GroupAwareAgent("AdaptiveDownloadFinderAgent", download_finder_prompt, download_finder_llm, download_finder_tools)

    # FileDownloadAgent
    file_download_prompt = os.path.join(current_dir, "FileDownloadAgent.prompt")
    file_download_llm = MistralLLM(model=LLMModel.MISTRAL_LARGE)
    file_download_prompt = PromptBuilder().from_file(file_download_prompt)
    file_download_tools = [FileDownloader()]
    file_download_agent = GroupAwareAgent("FileDownloadAgent", file_download_prompt, file_download_llm, file_download_tools)

    # DocumentAnalysisAgent
    doc_analysis_prompt = os.path.join(current_dir, "DocumentAnalysisAgent.prompt")
    doc_analysis_llm = ClaudeChatLLM(model=LLMModel.CLAUDE_3_5_SONNET)
    doc_analysis_prompt = PromptBuilder().from_file(doc_analysis_prompt)
    doc_analysis_agent = GroupAwareAgent("DocumentAnalysisAgent", doc_analysis_prompt, doc_analysis_llm, [])

    # EnglishWritingAgent
    english_writing_prompt = os.path.join(current_dir, "EnglishWritingAgent.prompt")
    english_writing_llm = GeminiLLM()
    english_writing_prompt = PromptBuilder().from_file(english_writing_prompt)
    english_writing_agent = GroupAwareAgent("EnglishWritingAgent", english_writing_prompt, english_writing_llm, [])

    # ChineseTranslationAgent
    chinese_translation_prompt = os.path.join(current_dir, "ChineseTranslationAgent.prompt")
    chinese_translation_llm = PerplexityLLM(LLMModel.LLAMA_3_1_SONAR_LARGE_128K_CHAT)
    chinese_translation_prompt = PromptBuilder().from_file(chinese_translation_prompt)
    chinese_translation_agent = GroupAwareAgent("ChineseTranslationAgent", chinese_translation_prompt, chinese_translation_llm, [])

    # Add agents to the orchestrator
    singleReplicaAgentOrchestrator.add_agent(pubmed_search_agent)
    singleReplicaAgentOrchestrator.add_agent(download_finder_agent)
    singleReplicaAgentOrchestrator.add_agent(file_download_agent)
    singleReplicaAgentOrchestrator.add_agent(doc_analysis_agent)
    singleReplicaAgentOrchestrator.add_agent(english_writing_agent)
    singleReplicaAgentOrchestrator.add_agent(chinese_translation_agent)

    # Set up CoordinatorAgent
    coordinator_llm = MistralLLM(model=LLMModel.MISTRAL_LARGE)
    coordinator_prompt = os.path.join(current_dir, "CoordinationAgent.prompt")
    coordinator_prompt = PromptBuilder().from_file(coordinator_prompt).set_variable_value(name="user_task", value="Write a Chinese article based on a PubMed research paper about recent advancements in cancer treatment")
    coordinator_tools = []

    coordinator_agent = CoordinatorAgent("CoordinationAgent", coordinator_prompt, coordinator_llm, coordinator_tools)
    singleReplicaAgentOrchestrator.set_coordinator_agent(coordinator_agent)

    return singleReplicaAgentOrchestrator

async def run_pubmed_article_workflow(agent_group: SingleReplicaAgentOrchestrator):
    result = await agent_group.run()
    print(result)

def main():
    setup_logger()
    agent_group = setup_agent_group()
    asyncio.run(run_pubmed_article_workflow(agent_group))

if __name__ == "__main__":
    main()