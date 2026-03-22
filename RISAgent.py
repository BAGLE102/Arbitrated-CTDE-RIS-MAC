class RISAgent:
    def __init__(self, ue_id, ris_num_elements):
        self.ue_id = ue_id
        self.ris_num_elements = ris_num_elements
        self.replay_buffer = ReplayBuffer()

    def select_action(self, obs):
        """
        回傳連續 request vector
        """
        return np.random.uniform(-1.0, 1.0, size=self.ris_num_elements)

    def store_transition(self, transition):
        self.replay_buffer.push(transition)

    def learn(self):
        pass
