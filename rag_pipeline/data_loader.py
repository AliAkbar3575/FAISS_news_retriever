# --------- importing necessary libraries ---------

from langchain_community.document_loaders import SeleniumURLLoader, WebBaseLoader
from langsmith import traceable


@traceable(name="data_loading")
def data_loading(urls):
    loader = WebBaseLoader(urls)
    data = loader.load()
    print("successfully loaded data!")
    return data