from agency_swarm.tools import BaseTool
from pydantic import Field
import json

class GetEnv(BaseTool):
    '''
    返回执行任务必要的环境信息。
    '''

    def run(self):
        # Mock
        env = {"endpoint": "ecs.cn-north-4.myhuaweicloud.com", "project_id": "05a86ccd57704ddbb96c66230646f286"}
        # env = {"endpoint": "ecs.cn-north-4.myhuaweicloud.com"}

        return json.dumps(obj=env, ensure_ascii=False)
        # return env
