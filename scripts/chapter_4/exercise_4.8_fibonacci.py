"""Simulate the discrete-time model:

    x_t = x_{t-1} + x_{t-2}, x_0 = 1, x_1 = 1
    
    Given that this is a second-order equation, we can convert it to a first order equation as follows:
    
    y_t = x_{t-1}
    
    such that
    
    x_t = x_{t-1} + y_{t-1}
"""

from dataclasses import dataclass, field

@dataclass
class Model:
    
    x1: float = 1.0
    y1: float = 1.0
    
    x: float = field(init=False)
    y: float = field(init=False)
    history: list[float] = field(init=False)
    
    def initialize(self) -> None:
        self.x = self.x1 # 1
        self.y = self.y1 # 1
        self.history = [self.y1, self.x1] # [1, 1]
        
    def observe(self) -> None:
        self.history.append(self.x)
        
    def update(self) -> None:
        x_new = self.x + self.y
        y_new = self.x
        
        self.x = x_new
        self.y = y_new         
        
    def run(self, steps: int) -> list[float]:
        self.initialize()
        for _ in range(steps):
            self.update()
            self.observe()
        return self.history
    
def main() -> None:
    model = Model()
    history = model.run(steps=10)
    print(history)
    
if __name__ == '__main__':
    main()
        
        
   