from agency_swarm import set_openai_key
from agency_swarm import Agent, Agency
from agency_swarm.tools import BaseTool

import os
from pydantic import Field

from agents.APIQueryGenerator import APIQueryGenerator
from agents.ParameterValueGenerator import ParameterValueGenerator
from agents.ArrayValueGenerator import ArrayValueGenerator

set_openai_key(os.getenv("OPENAI_API_KEY"))

GPT_4O = "gpt-4o-2024-08-06"
GPT_3_5_TURBO = "gpt-3.5-turbo"
GPT_4O_MINI = "gpt-4o-mini"

api_query_generator = APIQueryGenerator(model=GPT_4O)
parameter_value_generator = ParameterValueGenerator(model=GPT_4O)
array_value_generator = ArrayValueGenerator(model=GPT_4O)

agency = Agency(
                    [
                        api_query_generator,
                        # [parameter_value_generator, api_query_generator],
                        [parameter_value_generator, array_value_generator],
                        [array_value_generator, parameter_value_generator]
                    ],
                    # shared_instructions=''
                    temperature=0,
                    max_prompt_tokens=25000
                )

agency.demo_gradio()
# agency.run_demo()
