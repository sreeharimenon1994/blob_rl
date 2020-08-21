import numpy as np

class Memory:
    """docstring for Memory"""
    def __init__(self, input_dims, batch_size, mem_size_per_agent, n_blobs):
        super(Memory, self).__init__()
        self.mem_size = mem_size_per_agent * n_blobs
        self.batch_size = batch_size
        self.n_blobs = n_blobs
        self.n_prev = 3
        self.mem_cntr = 0

        self.state = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.new_state = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.action = np.zeros((self.mem_size, 3), dtype=np.int32)
        self.reward = np.zeros(self.mem_size, dtype=np.float32)
        self.stack_overflow_flag = False
        self.done = np.zeros(self.mem_size, dtype=np.bool)

    def store(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        if self.mem_cntr > self.n_prev:
            live_n_blobs = state.shape[0]
            if self.mem_cntr + live_n_blobs >= self.mem_size:
                self.mem_cntr = 0
                self.stack_overflow_flag = True
            
            self.state[self.mem_cntr: self.mem_cntr + live_n_blobs] = state
            self.action[self.mem_cntr: self.mem_cntr + live_n_blobs] = action
            self.new_state[self.mem_cntr: self.mem_cntr + live_n_blobs] = state_
            self.reward[self.mem_cntr: self.mem_cntr + live_n_blobs] = reward
            self.done[self.mem_cntr: self.mem_cntr + live_n_blobs] = done
            
            self.mem_cntr += live_n_blobs
        else:
            self.mem_cntr += 1


    def retrieve(self):
        if not self.stack_overflow_flag:
            max_mem = min(self.mem_cntr, self.mem_size)
        else:
            max_mem = self.mem_size

        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        state_batch = self.state[batch]
        new_state_batch = self.new_state[batch]
        action_batch = self.action[batch]
        reward_batch = self.reward[batch].reshape(-1, 1)
        done_batch = self.done[batch]
        return state_batch, new_state_batch, action_batch, reward_batch, done_batch

    def __len__(self):
        m = min(self.mem_cntr, self.mem_size)
        m = m if m > 0 else 0
        return m


class Prev_Observation:
    """docstring for Prev_Observation"""
    def __init__(self, n_prev=3):
        super(Prev_Observation, self).__init__()
        self.n_prev = n_prev
        self.prev = []

    def store(self, observation):
        self.prev.append(observation)

        if len(self.prev) == self.n_prev + 1:
            del self.prev[0]
        return self.prev

    @property
    def state(self):
        state = self.prev[0]
        for x in self.prev[1:]:
            state = np.append(state, x ,axis=1)
        state = state.astype(np.float32)
        return state

    def __len__(self):
        return len(self.prev)
    


        