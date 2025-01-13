from agency_swarm.tools import BaseTool
from pydantic import Field
import pandas as pd
import json
import re

from api_database import search_from_sqlite, API_DATABASE_FILE
from utils import try_parse_json

from tools.GetEnv import GetEnv
from tools.FillParameterTable import FillParameterTable

class FillAPI(BaseTool):
    '''
    填充一个 API 的所有参数值。
    '''

    user_requirement: str = Field(..., description="自然语言的用户需求")
    api_name: str = Field(..., description="目标API名")
    do_request: str = Field(..., description="是否执行API请求，为'true'时执行请求并返回响应，为'false'时返回API方法与参数值。")

    def run(self):
        print(f"debug: FillAPI BEGIN")

        # with open("fillapi-history.log", "a", encoding='utf-8')as f:
        #     f.write(f"FillAPI() call: user_requirement = {self.user_requirement}, api_name = {self.api_name}, do_request = {self.do_request}\n\n")
        #     f.close()

        # get environment information
        get_env_instance = GetEnv(from_tool = self)
        env_str = get_env_instance.run()
        print(f"debug: FillAPI - env_str = {env_str}")
        env = try_parse_json(env_str)
        print(f"debug: FillAPI - env = {env}")

        # get general information about this API
        api_info_df = search_from_sqlite(database_path=API_DATABASE_FILE, table_name='api_info', condition=f'name=\'{self.api_name}\'')
        print(f"debug: FillAPI - api_info_df = \n{api_info_df}")

        assert len(api_info_df) == 1
        api_info = api_info_df.iloc[0]
        print(f"debug: FillAPI - api_info = \n{api_info}")
        method = api_info.loc["method"]     # asuume no parameters
        uri = api_info.loc["uri"]           # assume some parameters

        print(f"debug: FillAPI - uri = {uri}")
        # uri = uri.format(endpoint = env["endpoint"])
        uri = re.sub(r'\{endpoint\}', env["endpoint"], uri)
        print(f"debug: FillAPI - uri = {uri}")

        # for each URI parameter, call Parameter Value Generator to decide its value
        uri_parameters_df = search_from_sqlite(database_path=API_DATABASE_FILE, table_name='uri_parameters', condition=f'api_name=\'{self.api_name}\'')
        uri_param_values = {}
        print(f"debug: FillAPI - uri_parameters_df = {uri_parameters_df}")
        for _, row in uri_parameters_df.iterrows():
            message_obj = {"user_requirement": self.user_requirement,
                           "env": env,
                           "api_name": self.api_name,
                           "parameter": row["parameter"],
                           "is_necessary": row["is_necessary"],
                           "description": row["description"]}
            print(f"debug: FillAPI - message_obj = {message_obj}")
            if row["type"] is not None:
                message_obj["type"] = row["type"]
            value = self.send_message_to_agent(recipient_agent_name="Parameter Value Generator", message=json.dumps(message_obj, ensure_ascii=False))
            if "不需要该参数" not in value:
                uri_param_values[row["parameter"]] = try_parse_json(value)

        # Call FillParameterTable() to decide the value of all request parameters
        fill_parameter_table_instance = FillParameterTable(from_tool = self,
                                                           user_requirement=self.user_requirement,
                                                           env=env_str,
                                                           api_name=self.api_name,
                                                           table_id=1) # assume root table is always table 1
        request_param_values_str = fill_parameter_table_instance.run()
        request_param_values = try_parse_json(request_param_values_str)

        # assemble the information and return
        info = {
            "method": method,
            "uri": uri,
            "uri_parameters": uri_param_values,
            "request_parameters": request_param_values
        }

        print(info)

        if self.do_request.lower() == "true":
            # Mock
            print("FillAPI: do_request")
            print(json.dumps(info, ensure_ascii=False, indent=4))
            ret = input("enter return value: ")
            return ret
        
        # with open("fillapi-history.log", "a", encoding='utf-8')as f:
        #     f.write(f"FillAPI() call: user_requirement = {self.user_requirement}, api_name = {self.api_name}, do_request = {self.do_request}, result:\n")
        #     f.write(f"{json.dumps(info, ensure_ascii=False)}\n\n")
        #     f.close()

        print(f"debug: FillAPI END")
        return json.dumps(info, ensure_ascii=False)
