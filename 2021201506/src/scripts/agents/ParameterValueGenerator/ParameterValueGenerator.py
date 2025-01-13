from agency_swarm.agents import Agent
from tools.FillParameterTable import FillParameterTable

class ParameterValueGenerator(Agent):
    def __init__(self, model: str = "gpt-4o-2024-08-06"):
        super().__init__(
            name="Parameter Value Generator",
            description="Parameter Value Generator 生成API请求中一个参数的值。",
            instructions="./instructions.md",
            tools=[FillParameterTable],
            temperature=0.3,
            max_prompt_tokens=25000,
            parallel_tool_calls=False,
            model=model
        )
        
    def response_validator(self, message):
        return message
