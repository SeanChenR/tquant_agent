from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv
load_dotenv()

model = ChatOpenAI(model="gpt-4o", temperature=0)

def custom_parser(message) -> str:
	html_code = message.content.split("```")[1].replace("html", "", 1)
	return html_code

def email_chain(results):
    print("撰寫 Email 內文")

    template = """
    作為Email文案撰寫者，您的任務是根據提供的回測結果，創建清晰且具規範性的電子郵件。您的角色至關重要，確保我們的溝通內容專業、有結構且視覺上吸引人。

    操作指示：
    - 初始標題：標題設計為一目均衡表策略的回測結果，電子郵件開頭包括使用者給定的回測條件，包括：回測區間、選擇的股票(全數列出)、比較標的。
    - 研究摘要：撰寫詳細的研究結果，包含圖片的詳細描述(圖片標題使用英文即可，不需要翻成中文)，相關回測指標的部分需做成表格，以上兩個一定都要！！！
    - HTML 格式化：使用適當的 HTML 元素（如標題、段落、項目符號）來增強可讀性，並且有圖片網址的需嵌入圖片在對應的圖片描述中。
    - 研究結果：需將研究的結果，也就是所有內容的大結論寫在最後。
    - 設計美感：需美化整個 html 設計，像是圖片與描述文字需要至於中央、背景設置為淺藍色、顯而易見的結論，不要只是單純的黑白色系。

    以下是回測結果的所有內容：{input}
    """

    prompt = PromptTemplate.from_template(template)

    chain = prompt | model | RunnableLambda(custom_parser)

    with open("backtest_stats.txt", "r") as f:
        content = f.read()
    
    final = content + "\n總結：" + results['output']

    html_content = chain.invoke({"input": final})

    subject_template = "請根據{input}回測結果想一個 Email 的寄件標題，回傳結果僅包含寄件標題即可，標題中需包含策略的名稱。"
    print("撰寫 Email 標題")
    subject_prompt = PromptTemplate.from_template(subject_template)
    subject_chain = subject_prompt | model
    subject = subject_chain.invoke({"input": final})

    mail_results = {"subject" : subject.content, "html_content" : html_content}
    return mail_results