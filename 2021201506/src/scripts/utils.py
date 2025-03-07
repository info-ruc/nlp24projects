import json

def try_parse_json(value: str):
    try:
        data = json.loads(value)
        assert isinstance(data, list) or isinstance(data, dict) or isinstance(data, str)
        value_parsed = data
    except (json.JSONDecodeError, AssertionError) as e:
        value_parsed = value
    
    return value_parsed