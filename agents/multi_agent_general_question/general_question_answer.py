import asyncio
import os
import logging
from autobyteus.agent.orchestrator.single_replica_agent_orchestrator import SingleReplicaAgentOrchestrator
from autobyteus.agent.group.group_aware_agent import GroupAwareAgent
from autobyteus.agent.group.coordinator_agent import CoordinatorAgent
from autobyteus.tools.browser.standalone.webpage_reader import WebPageReader, CleaningMode
from autobyteus.llm.models import LLMModel
from autobyteus.llm.rpa.gemini_llm import GeminiLLM
from autobyteus.llm.rpa.claudechat_llm import ClaudeChatLLM
from autobyteus.llm.rpa.perplexity_llm import PerplexityLLM
from autobyteus.llm.rpa.groq_llm import GroqLLM
from autobyteus.llm.rpa.mistral_llm import MistralLLM
from autobyteus.prompt.prompt_builder import PromptBuilder
from autobyteus.tools.browser.standalone.google_search_ui import GoogleSearch
from autobyteus.tools.browser.standalone.webpage_reader import WebPageReader

def setup_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Create handlers
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler('agent_workflow.log')

    # Set levels for handlers
    console_handler.setLevel(logging.INFO)
    file_handler.setLevel(logging.DEBUG)

    # Create formatters and add them to handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

def setup_agent_group():
    singleReplicaAgentOrchestrator = SingleReplicaAgentOrchestrator()

    # Set up GoogleSearchAgent
    current_dir = os.path.dirname(os.path.abspath(__file__))
    google_search_prompt = os.path.join(current_dir, "google_search_agent.prompt")
    webpage_analyzer_llm = PerplexityLLM(LLMModel.LLAMA_3_1_70B_INSTRUCT) #GroqLLM(model=LLMModel.LLAMA_3_1_70B_VERSATILE) # ) ## #
    google_search_prompt = PromptBuilder().from_file(google_search_prompt)
    google_search_tools = [GoogleSearch()]
    google_search_agent = GroupAwareAgent("GoogleSearchAgent", google_search_prompt, webpage_analyzer_llm, google_search_tools)

    # Set up WebAnalyzerAgent
    webpage_analyzer_prompt = os.path.join(current_dir, "webpage_analyzer_agent_v2.prompt")
    webpage_analyzer_llm = PerplexityLLM(LLMModel.LLAMA_3_1_70B_INSTRUCT) # # PerplexityLLM(LLMModel.LLAMA_3_1_SONAR_LARGE_128K_CHAT) #MistralLLM(model=LLMModel.MISTRAL_LARGE)#ClaudeChatLLM(model=LLMModel.CLAUDE_3_5_SONNET) ## # #GroqLLM(model=LLMModel.LLAMA_3_1_70B_VERSATILE) ##MistralLLM(model=LLMModel.MISTRAL_LARGE) #
    webpage_analyzer_prompt = PromptBuilder().from_file(webpage_analyzer_prompt)
    webpage_reader_tools = [WebPageReader(cleaning_mode=CleaningMode.TEXT_CONTENT_FOCUSED)]
    webpage_analyzer_agent = GroupAwareAgent("WebContentAnalysisAgent", webpage_analyzer_prompt, webpage_analyzer_llm, webpage_reader_tools)

    # Set up SummaryAgent
    #summary_llm = GeminiLLM()
    #summary_prompt = os.path.join(current_dir, "summary_agent.prompt")
    #summary_prompt = PromptBuilder().from_file(summary_prompt)
    #summary_agent = GroupAwareAgent("SummarizationAgent", summary_prompt, summary_llm, [])

    # Add agents to the group
    singleReplicaAgentOrchestrator.add_agent(google_search_agent)
    singleReplicaAgentOrchestrator.add_agent(webpage_analyzer_agent)
    #singleReplicaAgentOrchestrator.add_agent(summary_agent)

    # Set up CoordinationAgent
    coordinator_llm = MistralLLM(model=LLMModel.MISTRAL_LARGE) # #
    #coordinator_prompt = PromptBuilder().from_file(coordinator_prompt).set_variable_value(name="user_task", value=
    #'''
    #I need to do some experiements on my computer for NGS technology to do some experiements. But i dont know how to do that. Give me back 
    #some instructionts, and perhaps some codes i do experiements with.
    #''')
    coordinator_prompt = os.path.join(current_dir, "coordinator_agent_v1.prompt")
    coordinator_prompt = PromptBuilder().from_file(coordinator_prompt).set_variable_value(name="user_task", value=
    '''
     
    ''')
    coordinator_tools = []  # The coordinator will use the SendMessageTo tool added by GroupAwareAgent


    coordinator_agent = CoordinatorAgent("CoordinationAgent", coordinator_prompt, coordinator_llm, coordinator_tools)
    singleReplicaAgentOrchestrator.set_coordinator_agent(coordinator_agent)

    return singleReplicaAgentOrchestrator

async def run_research_summary_workflow(agent_group: SingleReplicaAgentOrchestrator):
    # Run the workflow
    result = await agent_group.run()
    # Print the result
    print(result)

def main():
    setup_logger()
    agent_group = setup_agent_group()
    asyncio.run(run_research_summary_workflow(agent_group))

if __name__ == "__main__":
    main()
