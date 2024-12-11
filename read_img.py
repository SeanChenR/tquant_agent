from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

image_model = ChatOpenAI(model_name="gpt-4o", temperature=0.5)

def read_img(results):
    with open("backtest_img_url.txt", "r") as f:
        urls = f.readlines()

    for url in urls:
        index = url.split('/')[-1].replace('.png', '').strip()
        message = [HumanMessage(
            content=[
                {"type": "text", "text": f"用繁體中文解釋圖片並做個小結論，這是使用一目均衡表策略的回測圖表，指標是{index}"},
                {
                    "type": "image_url", 
                    "image_url": {
                        "url": url.strip()
                    }
                },
            ]
        )]

        result = image_model.invoke(message)
        text = f"{index}:圖片網址：{url}\n{result.content}"
        print(text)

        with open("backtest_stats.txt", "a") as f:
            f.write(f"\n{text}\n")

    return results