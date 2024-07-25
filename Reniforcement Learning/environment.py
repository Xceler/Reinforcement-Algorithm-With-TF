class SimpleEnvironment:
    def __init__(self):
        self.state = 0 
        self.end_state = 5 
    
    def reset(self):
        self.state = 0
        return self.state 
    
    def step(self, action):
        if action == 1:
            self.state += 1
        else:
            self.state -= 1
        
        if self.state == self.end_state:
            return self.state, 1, True 
        
        elif self.state < 0 or self.state > self.end_state:
            return self.state, -1, True
        
        else:
            return self.state, 0, False 
        
    def render(self):
        print(f"State: {self.state}")