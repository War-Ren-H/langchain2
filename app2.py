from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
import os

os.environ["OPEN_API_KEY"] = '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'


llm = OpenAI(temperature=0.9)
prompt = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?",
)

from langchain.chains import LLMChain
chain = LLMChain(llm=llm, prompt=prompt)

print(chain.run("colorful socks"))