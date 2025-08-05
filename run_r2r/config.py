import pydantic

class APIConfig(pydantic.BaseModel, extra="allow"):
    API_KEY: str
    BASE_URL: str

class VLMConfig(pydantic.BaseModel, extra="allow"):
    model_name: str = "Qwen/Qwen2.5-VL-72B-Instruct"
    type: str = "local" # server or local
    port: int = 8001
    chat_single_node: bool = True

    # API
    QWen_API: APIConfig = APIConfig(
        API_KEY="",
        BASE_URL=""
    )
    OpenAI_API: APIConfig = APIConfig(
        API_KEY="",
        BASE_URL=""
    )