from langtrain import train
import os
from dotenv import load_dotenv

load_dotenv()

docsearch = train(["https://portfolio-zeta-peach-28.vercel.app/"])


from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

llm = OpenAI(temperature=0, openai_api_key= os.getenv('OPENAI_API_KEY'))
chain = load_qa_chain(llm, chain_type="stuff")

query = "What is the companies this person has worked at?"
docs = docsearch.similarity_search(query, include_metadata=True)

print(chain.run(input_documents=docs, question=query))
