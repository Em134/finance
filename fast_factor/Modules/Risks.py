from typing import Any
from .BaseModules import BaseModule


class BaskRisk(BaseModule):
    def __init__(self) -> None:
        super().__init__()
        pass
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return args, kwds
