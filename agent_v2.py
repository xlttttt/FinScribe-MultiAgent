import os
import yfinance as yf
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
# 引入 LangGraph 的预构建智能体
from langgraph.prebuilt import create_react_agent

# 1. 基础配置
load_dotenv()
os.environ["OPENAI_API_BASE"] = "https://api.deepseek.com"

# 2. 定义工具 (和之前完全一样)
@tool
def get_stock_price(ticker: str) -> str:
    """
    获取指定美股代码的最新实时收盘价。
    输入参数必须是标准股票代码（如 AAPL，TSLA，NVDA）。
    """
    print(f"\n[🔧 工具执行中] 正在后台联网查询 {ticker} 的股价...")
    try:
        stock = yf.Ticker(ticker)
        price = stock.history(period="1d")['Close'].iloc[-1]
        return f"系统查到的 {ticker} 最新收盘价为 {price:.2f} 美元"
    except Exception as e:
        return f"查询失败，错误: {e}"

tools = [get_stock_price]

# 3. 初始化大模型
llm = ChatOpenAI(
    model="deepseek-chat", 
    temperature=0,
    api_key=os.getenv("DEEPSEEK_API_KEY") 
)

# 4. 🎭 见证新一代架构的优雅：一键创建基于图（Graph）的 Agent
# create_react_agent 在底层自动帮你画好了一张包含“思考->调用->总结”的状态机网络图
agent_executor = create_react_agent(llm, tools)

# 5. 发起测试挑战
query = "帮我查一下苹果公司今天的股价是多少？"
print(f"👤 用户的提问: {query}\n")

# LangGraph 的数据格式规范：必须传入一个包含消息记录的字典
result = agent_executor.invoke({"messages": [("user", query)]})

# 6. 解析并打印最终结果
print("\n🤖 最终给用户的自然语言回答:")
# result["messages"] 记录了整个流转过程的所有消息，最后一条就是模型的最终回复
print(result["messages"][-1].content)