from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
from langchain_core.messages import SystemMessage,HumanMessage
from parser import Info
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_agent
load_dotenv()

class Gemini_Client:
    def __init__(self):
        self.model = ChatGoogleGenerativeAI(model="gemini-2.5-flash",api_key=os.getenv("GEMINI_API_KEY"))
    
    def complete(self,system_prompt,user_prompt):
        response = self.model.invoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]         
        )
        return response.text

    def stream(self,system_prompt,user_prompt):
        response = self.model.stream(
            [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]         
        )
        return response

    def structure_output(self,system_prompt,user_prompt):
        agent = create_agent(
            model="google_genai:gemini-2.5-flash",
            api_key=os.getenv("GEMINI_API_KEY"),
            response_format=Info,
            system_prompt=system_prompt,
        )
        response = agent.invoke({"user_prompt":user_prompt})
        return response
