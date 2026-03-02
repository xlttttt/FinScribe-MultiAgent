import os
import yfinance as yf
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
# 引入 Tavily 官方库
from tavily import TavilyClient

# 1. 基础配置
load_dotenv()
os.environ["OPENAI_API_BASE"] = "https://api.deepseek.com"

# 2. 工具 A：查股价 (我们上一版的杰作)
@tool
def get_stock_price(ticker: str) -> str:
    """
    获取指定美股代码的最新实时收盘价。
    输入参数必须是标准股票代码（如 AAPL，TSLA，NVDA）。
    """
    print(f"\n[📈 动作] 正在查询 {ticker} 的股价...")
    try:
        stock = yf.Ticker(ticker)
        price = stock.history(period="1d")['Close'].iloc[-1]
        return f"{ticker} 最新收盘价为 {price:.2f} 美元"
    except Exception as e:
        return f"查询失败，错误: {e}"

# 3. 💥 新增工具 B：全网搜新闻
@tool
def search_latest_news(query: str) -> str:
    """
    搜集最新的全网新闻、行业动态和公司基本面信息。
    当用户询问某家公司的最新消息、大事件或未知的商业动态时调用此工具。
    """
    print(f"\n[🌐 动作] 正在全网搜索: {query} ...")
    try:
        # 初始化 Tavily 客户端
        tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        # 执行基础搜索，只提取前3条最相关的结果防止上下文超载
        response = tavily_client.search(query, search_depth="basic", max_results=3)
        
        # 提取搜索结果中的精简摘要
        news_summaries = [f"- {res['title']}: {res['content']}" for res in response['results']]
        return "\n".join(news_summaries)
    except Exception as e:
        return f"搜索失败，错误: {e}"

# 4. 把两个工具打包放入工具箱
tools = [get_stock_price, search_latest_news]

# 5. 初始化大模型与 Agent
llm = ChatOpenAI(
    model="deepseek-chat", 
    temperature=0,
    api_key=os.getenv("DEEPSEEK_API_KEY") 
)
agent_executor = create_react_agent(llm, tools)

# 6. 💥 终极混合测试挑战！
# 注意这个提问，它故意包含了两个不同的意图
query = "查一下特斯拉（Tesla）今天的股价，并搜一下马斯克（Elon Musk）最近几天有什么大新闻？最后帮我总结成一段话。"
print(f"👤 用户的提问: {query}\n")

# 启动！
result = agent_executor.invoke({"messages": [("user", query)]})

print("\n🤖 最终给用户的自然语言总结:")
print(result["messages"][-1].content)