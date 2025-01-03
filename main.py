import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_gigachat.chat_models import GigaChat
from langchain_community.tools import DuckDuckGoSearchRun, tool
from langchain.agents import AgentExecutor, create_gigachat_functions_agent

dotenv_path = Path(__file__).resolve().parent / '.env'
load_dotenv(dotenv_path)
giga_token = os.getenv('GIGACHAT_AUTH_KEY')

giga = GigaChat(
    credentials=giga_token,
    model='GigaChat',
    verify_ssl_certs=False
)

search_tool = DuckDuckGoSearchRun()
tools = [search_tool]

agent = create_gigachat_functions_agent(giga, tools)

# AgentExecutor создает среду, в которой будет работать агент
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
)

result = agent_executor.invoke(
    {
        'input': "Найди текущий курс доллара к рублю и напечатай только число"
    }
)['output']

print(f'{result = }')