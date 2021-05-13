class Observation:
    def __init__(self, data, valids, _type):
        self.data = data
        self.type = _type
        self.valids = valids
    
    def __repr__(self):
        return self.type

class BaseTFObservation:
    def __init__(self, obs):
        self.convert(obs)
    
    def convert(self, obs):
        raise NotImplementedError