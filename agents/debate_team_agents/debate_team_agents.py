# File: autobyteus/agents/debate_team_agents.py

import asyncio
import os
import logging
from autobyteus.agent.orchestrator.single_replica_agent_orchestrator import SingleReplicaAgentOrchestrator
from autobyteus.agent.group.group_aware_agent import GroupAwareAgent
from autobyteus.agent.group.coordinator_agent import CoordinatorAgent
from autobyteus.llm.models import LLMModel
from autobyteus.llm.rpa.claudechat_llm import ClaudeChatLLM
from autobyteus.prompt.prompt_builder import PromptBuilder
from autobyteus.tools.ask_user_input import AskUserInput
from autobyteus.tools.timer import Timer
from autobyteus.llm.rpa.perplexity_llm import PerplexityLLM

from autobyteus.tools.browser.session_aware.browser_session_aware_webpage_reader import BrowserSessionAwareWebPageReader
from autobyteus.tools.browser.session_aware.browser_session_aware_web_element_trigger import BrowserSessionAwareWebElementTrigger
from autobyteus.tools.browser.session_aware.browser_session_aware_webpage_screenshot_taker import BrowserSessionAwareWebPageScreenshotTaker
from autobyteus.tools.browser.session_aware.browser_session_aware_navigate_to import BrowserSessionAwareNavigateTo

from autobyteus.tools.browser.standalone.google_search_ui import GoogleSearch
from autobyteus.tools.browser.standalone.webpage_reader import WebPageReader, CleaningMode

from autobyteus.llm.rpa.gemini_llm import GeminiLLM
from autobyteus.llm.rpa.claudechat_llm import ClaudeChatLLM
from autobyteus.llm.rpa.perplexity_llm import PerplexityLLM
from autobyteus.llm.rpa.groq_llm import GroqLLM
from autobyteus.llm.rpa.mistral_llm import MistralLLM

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

def create_debate_teams(team_a_id, team_b_id):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    debate_team_dir = os.path.join(current_dir)

    singleReplicaAgentOrchestrator = SingleReplicaAgentOrchestrator()

    for team_id in [team_a_id, team_b_id]:
        # FactChecker
        fact_checker_prompt = os.path.join(debate_team_dir, "FactChecker.prompt")
        fact_checker_llm = PerplexityLLM(LLMModel.LLAMA_3_1_70B_INSTRUCT)#ClaudeChatLLM(model=LLMModel.CLAUDE_3_5_SONNET)
        fact_checker_prompt = PromptBuilder().from_file(fact_checker_prompt).set_variable_value(name="agent_id", value=f"{team_id}FactChecker-001")
        fact_checker_agent = GroupAwareAgent(f"{team_id}FactChecker", fact_checker_prompt, fact_checker_llm, [])
        singleReplicaAgentOrchestrator.add_agent(fact_checker_agent)

        # RebuttalSpecialist
        rebuttal_specialist_prompt = os.path.join(debate_team_dir, "RebuttalSpecialist.prompt")
        rebuttal_specialist_llm = PerplexityLLM(LLMModel.LLAMA_3_1_70B_INSTRUCT)
        rebuttal_specialist_prompt = PromptBuilder().from_file(rebuttal_specialist_prompt).set_variable_value(name="agent_id", value=f"{team_id}RebuttalSpecialist-001")
        rebuttal_specialist_agent = GroupAwareAgent(f"{team_id}RebuttalSpecialist", rebuttal_specialist_prompt, rebuttal_specialist_llm, [])
        singleReplicaAgentOrchestrator.add_agent(rebuttal_specialist_agent)

        # Researcher
        researcher_prompt = os.path.join(debate_team_dir, "Researcher.prompt")
        researcher_llm = PerplexityLLM(LLMModel.LLAMA_3_1_70B_INSTRUCT)
        researcher_prompt = PromptBuilder().from_file(researcher_prompt).set_variable_value(name="agent_id", value=f"{team_id}Researcher-001")
        researcher_agent = GroupAwareAgent(f"{team_id}Researcher", researcher_prompt, researcher_llm, [])
        singleReplicaAgentOrchestrator.add_agent(researcher_agent)

        # TeamCaptain
        team_captain_prompt = os.path.join(debate_team_dir, "TeamCaptain.prompt")
        team_captain_llm = PerplexityLLM(LLMModel.LLAMA_3_1_70B_INSTRUCT)
        team_captain_prompt = PromptBuilder().from_file(team_captain_prompt).set_variable_value(name="agent_id", value=f"{team_id}Captain-001")
        team_captain_agent = GroupAwareAgent(f"{team_id}Captain", team_captain_prompt, team_captain_llm, [])
        singleReplicaAgentOrchestrator.add_agent(team_captain_agent)

    google_search_prompt = os.path.join(current_dir, "google_search_agent.prompt")
    webpage_analyzer_llm = GroqLLM(model=LLMModel.LLAMA_3_1_70B_VERSATILE) # PerplexityLLM(LLMModel.LLAMA_3_1_405B_REASONING) ## #
    google_search_prompt = PromptBuilder().from_file(google_search_prompt)
    google_search_tools = [GoogleSearch()]
    google_search_agent = GroupAwareAgent("GoogleSearchAgent", google_search_prompt, webpage_analyzer_llm, google_search_tools)

    # Set up WebAnalyzerAgent
    webpage_analyzer_prompt = os.path.join(current_dir, "webpage_analyzer_agent.prompt")
    webpage_analyzer_llm = PerplexityLLM(LLMModel.LLAMA_3_1_70B_INSTRUCT) # # PerplexityLLM(LLMModel.LLAMA_3_1_SONAR_LARGE_128K_CHAT) #MistralLLM(model=LLMModel.MISTRAL_LARGE)#ClaudeChatLLM(model=LLMModel.CLAUDE_3_5_SONNET) ## # #GroqLLM(model=LLMModel.LLAMA_3_1_70B_VERSATILE) ##MistralLLM(model=LLMModel.MISTRAL_LARGE) #
    webpage_analyzer_prompt = PromptBuilder().from_file(webpage_analyzer_prompt)
    webpage_reader_tools = [WebPageReader(cleaning_mode=CleaningMode.TEXT_CONTENT_FOCUSED)]
    webpage_analyzer_agent = GroupAwareAgent("WebContentAnalysisAgent", webpage_analyzer_prompt, webpage_analyzer_llm, webpage_reader_tools)

    singleReplicaAgentOrchestrator.add_agent(google_search_agent)
    singleReplicaAgentOrchestrator.add_agent(webpage_analyzer_agent)

    # CoordinatorAgent
    coordinator_prompt = os.path.join(debate_team_dir, "DebateCoordinator.prompt")
    coordinator_llm = ClaudeChatLLM(model=LLMModel.CLAUDE_3_5_SONNET)
    coordinator_prompt = PromptBuilder().from_file(coordinator_prompt)
    coordinator_agent = CoordinatorAgent("DebateCoordinator", coordinator_prompt, coordinator_llm, [
        AskUserInput(),Timer()
    ])
    singleReplicaAgentOrchestrator.set_coordinator_agent(coordinator_agent)

    return singleReplicaAgentOrchestrator

def setup_agent_groups():
    return create_debate_teams("TeamA", "TeamB")

async def run_debate_workflow(orchestrator):
    result = await orchestrator.run()
    print(result)

def main():
    setup_logger()
    orchestrator = setup_agent_groups()
    asyncio.run(run_debate_workflow(orchestrator))

if __name__ == "__main__":
    main()