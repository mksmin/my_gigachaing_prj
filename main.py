import os
import pyfiglet

from pathlib import Path
from dotenv import load_dotenv

from langchain_gigachat.chat_models import GigaChat
from langchain_community.tools import DuckDuckGoSearchRun, tool
from langchain.agents import AgentExecutor, create_gigachat_functions_agent
from langchain_core.messages import AIMessage, HumanMessage

dotenv_path = Path(__file__).resolve().parent / '.env'
load_dotenv(dotenv_path)
giga_token = os.getenv('GIGACHAT_AUTH_KEY')

giga = GigaChat(
    credentials=giga_token,
    model='GigaChat',
    verify_ssl_certs=False
)


@tool
def draw_banner(number: str) -> str:
    """ Рисует баннер с текстом результатов кода в виде Ascii-графики

    :param number: Число, которое нужно нарисовать на баннере
    :return:
    """
    pyfiglet.print_figlet(number, font='epic')
    return 'Draw complete'


search_tool = DuckDuckGoSearchRun()
tools = [search_tool]
new_tools = [search_tool, draw_banner]

agent = create_gigachat_functions_agent(giga, new_tools)

# AgentExecutor создает среду, в которой будет работать агент
agent_executor = AgentExecutor(
    agent=agent,
    tools=new_tools,
    verbose=False
)

# result = agent_executor.invoke(
#     {
#         'input': "Найди в интернете текущий курс доллара к рублю и нарисуй это число на банере"
#     }
# )['output']

# pyfiglet.print_figlet('Hello!', font='epic')
# print(f'{result = }')

# result = agent_executor.invoke(
#     {
#         'chat_history': [
#             HumanMessage(
#                 content='Привет! Запомни трех животных - слон, жираф, крокодил'
#             ),
#             AIMessage(content='Привет! Хорошо, я запомнил')
#         ],
#         'input': 'Что я просил тебя запомнить?'
#     }
# )['output']
# print(f'{result = }')

chat_history = []
while True:
    user_input = input('Вы: ')
    print(f'Пользователь: {user_input}')
    if user_input == '':
        break

    result = agent_executor.invoke(
        {
            "chat_history": chat_history,
            'input': user_input
        }
    )
    chat_history.append(HumanMessage(content=user_input))
    chat_history.append(AIMessage(content=result['output']))
    print(f'Агент: {result["output"]}')
