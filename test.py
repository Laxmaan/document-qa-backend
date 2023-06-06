import os
os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_VERSION"] = "2023-03-15-preview"
os.environ["OPENAI_API_BASE"] = "https://idocopenaigpt.openai.azure.com/"
os.environ["OPENAI_API_KEY"] = "95776649ac7a4b048c834003fd315264"

from langchain.llms import AzureOpenAI

llm = AzureOpenAI(
    deployment_name="idocgptmodels",
        model_name="text-davinci-003",
)


print(llm("tell me about google"))