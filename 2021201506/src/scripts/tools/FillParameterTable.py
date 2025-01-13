from agency_swarm.tools import BaseTool
from pydantic import Field
import pandas as pd
import json

from api_database import search_from_sqlite, API_DATABASE_FILE
from utils import try_parse_json

class FillParameterTable(BaseTool):
    '''
    填充一个 API 在一张参数表中的所有参数值。
    '''

    user_requirement: str = Field(..., description="自然语言的用户需求")
    env: str = Field(..., description="环境信息")
    api_name: str = Field(..., description="目标API名")
    table_id: int = Field(..., description="表号，常见于“详情请参见表...”，默认值为0")

    def run(self):

        # with open("fillparametertable-history.log", "a", encoding='utf-8')as f:
        #     f.write(f"FillParameterTable() call: user_requirement = {self.user_requirement}, env = {self.env}, api_name = {self.api_name}, table_id = {self.table_id}\n\n")
        #     f.close()

        # for each parameter in this table, call Parameter Value Generator to decide its value.
        param_table_df = search_from_sqlite(database_path=API_DATABASE_FILE, table_name='request_parameters', condition=f"api_name='{self.api_name}' AND table_id='{self.table_id}'")
        param_values = {}

        for _, row in param_table_df.iterrows():
            message_obj = {"user_requirement": self.user_requirement,
                "env": self.env,
                "api_name": self.api_name,
                "parameter": row["parameter"],
                "is_necessary": row["is_necessary"],
                "description": row["description"]}
            if row["type"] is not None:
                message_obj["type"] = row["type"]
            value_str = self.send_message_to_agent(recipient_agent_name="Parameter Value Generator", message=json.dumps(message_obj, ensure_ascii=False))
            if "不需要该参数" not in value_str:
                param_values[row["parameter"]]= try_parse_json(value_str)
        
            # with open("fillparametertable-history.log", "a", encoding='utf-8')as f:
            #     f.write(f"FillParameterTable() call: user_requirement = {self.user_requirement}, env = {self.env}, api_name = {self.api_name}, table_id = {self.table_id}, result:\n")
            #     f.write(f"{json.dumps(param_values, ensure_ascii=False)}\n\n")
            #     f.close()

        return json.dumps(param_values, ensure_ascii=False)
