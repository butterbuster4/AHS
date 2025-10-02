class Tile():
    def __init__(self, image, index):
        self.image = image
        self.index = index
        self.right = None
        self.left = None
        self.up = None
        self.down = None
        
    def __eq__(self, other):
        return isinstance(other, Tile) and self.index == other.index

    def __hash__(self):
        return hash(self.index)