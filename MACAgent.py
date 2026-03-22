class MACAgent:
    def __init__(self, ue_id, num_channels):
        self.ue_id = ue_id
        self.num_channels = num_channels
        self.replay_buffer = ReplayBuffer()

    def select_action(self, obs):
        """
        回傳:
        0 -> Listen
        1 ~ num_channels -> Transmit on channel n
        """
        queue_len = obs["queue_len"]

        if queue_len == 0:
            return 0

        # 簡單策略：大部分時間 transmit，少部分 listen
        if random.random() < 0.2:
            return 0
        return random.randint(1, self.num_channels)

    def store_transition(self, transition):
        self.replay_buffer.push(transition)

    def learn(self):
        pass
