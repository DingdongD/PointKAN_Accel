class cache:
    def __init__(self):
        self.entries_num = 128
        self.content = []
        self.counter = [0 for i in range(self.entries_num)]
    
    def get(self, value):
        if value in self.content:
            self.counter[self.content.index(value)] = 0
            return True     # cache hit
        else:
            return False    # cache miss
    
    def put(self, value):
        if len(self.content) < self.entries_num:
            self.content.append(value)
        else:
            # cache replace in LRU policy
            idx = self.counter.index(max(self.counter))
            # if self.content[idx] == 367:
            #     print('367 is replcaed from cache')
            self.content[idx] = value
            self.counter[idx] = 0
    
    def clean(self):
        self.content = []
        self.counter = [0 for i in range(self.entries_num)]
    
    def cycle(self):
        for i in range(self.entries_num):
            if i < len(self.content):
                self.counter[i] += 1
            else:
                break
        # self.counter = [self.counter[i]+1 for i in range(self.counter) if i<len(self.content)]