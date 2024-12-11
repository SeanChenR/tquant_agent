from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda
from backtest_tools import Ichimoku_Kinko_Hyo
from read_img import read_img
from email_content import email_chain
from send_mail import send_email
from dotenv import load_dotenv
load_dotenv()

chat_model = ChatOpenAI(model="gpt-4o", temperature=0)

tools = [Ichimoku_Kinko_Hyo]

prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that you can run the backtesting."),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ]
)

agent = create_openai_functions_agent(
    llm=chat_model,
    prompt=prompt,
    tools=tools,
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
)

agent_chain = agent_executor | RunnableLambda(read_img) | RunnableLambda(email_chain) | RunnableLambda(send_email)

response = agent_chain.invoke({"input": "先幫我進行選股，回測區間為 2019-04-01 至 2024-04-01，選股的id為 'IX0002'，最後幫我將回測結果的指標做一個詳細結論。"})