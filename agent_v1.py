import os
import yfinance as yf
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

# 1. 加载 .env 文件中的 API Key (保护你的资产)
load_dotenv()

# 将 OpenAI 的基础地址重定向到 DeepSeek 的服务器
os.environ["OPENAI_API_BASE"] = "https://api.deepseek.com"

# 2. 定义工具 (Tool)：这是 Agent 的“手”
@tool
def get_stock_price(ticker: str) -> str:
    """
    获取指定美股代码的最新实时收盘价。
    输入参数必须是标准股票代码（如 AAPL 代表苹果，TSLA 代表特斯拉，NVDA 代表英伟达）。
    """
    print(f"\n[🔧 工具执行中] 正在后台联网查询 {ticker} 的股价...")
    try:
        stock = yf.Ticker(ticker)
        # 获取最近一天的收盘价
        price = stock.history(period="1d")['Close'].iloc[-1]
        return f"{ticker} 的最新收盘价为 {price:.2f} 美元"
    except Exception as e:
        return f"查询失败，请检查股票代码是否正确。错误: {e}"

# 3. 初始化大模型：这是 Agent 的“大脑”
# temperature=0 表示让模型输出尽可能确定和严谨，不瞎发散
llm = ChatOpenAI(
    model="deepseek-chat", 
    temperature=0,
    api_key=os.getenv("DEEPSEEK_API_KEY") 
)

# 4. 关键动作：把工具绑定给大脑
tools = [get_stock_price]
llm_with_tools = llm.bind_tools(tools)

# 5. 发起测试挑战！
query = "帮我查一下英伟达（Nvidia）今天的股价是多少？"
print(f"👤 用户的提问: {query}")
print("🧠 大模型正在思考...")

# 让大模型处理这个问题
response = llm_with_tools.invoke(query)

# 6. 解析大模型的决定
print("\n--- 大模型的判断结果 ---")
if response.tool_calls:
    print("🎯 大模型判断：这个问题超出了我的内部知识，我必须调用外部工具！")
    for tool_call in response.tool_calls:
        print(f"它决定调用的函数名: {tool_call['name']}")
        print(f"它从自然语言中提取的参数: {tool_call['args']}")
else:
    print("💬 模型认为不需要外部工具，直接回答:")
    print(response.content)