from langchain.llms import OpenAI
import os

os.environ["OPEN_API_KEY"] = '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'

llm = OpenAI(temperature = 0.9)

text = "What would be a good company name for a company that makes colorful socks?"

print(llm(text))