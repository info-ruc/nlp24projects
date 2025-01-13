from agency_swarm.agents import Agent
from tools.FillAPI import FillAPI

class APIQueryGenerator(Agent):
    def __init__(self, model: str = "gpt-4o-2024-08-06"):
        super().__init__(
            name="API Query Generator",
            description="API Query Generator 从自然语言需求生成格式化的API请求。",
            instructions="./instructions.md",
            tools=[FillAPI],
            temperature=0.3,
            max_prompt_tokens=25000,
            parallel_tool_calls=False,
            model=model
        )
        
    def response_validator(self, message):
        return message
