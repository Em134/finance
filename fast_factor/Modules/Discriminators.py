from typing import Any
from .BaseModules import BaseModule


class BaseDiscriminator(BaseModule):
    def __init__(self, 
                 alpha: BaseModule=None, 
                 risk: BaseModule=None, 
                 costs: BaseModule=None, 
                 portfolio: BaseModule=None, 
                 executor: BaseModule=None
                 ) -> None:
        
        super().__init__()
        self.alpha = alpha
        self.risk = risk
        self.costs = costs
        self.portfolio = portfolio
        self.executor = executor
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass