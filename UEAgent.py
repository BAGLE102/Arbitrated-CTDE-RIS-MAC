class UEAgent:
    def __init__(self, ue_id, config: Config):
        self.id = ue_id
        self.cfg = config

        self.mac_agent = MACAgent(ue_id, config.num_channels)
        self.ris_agent = RISAgent(ue_id, config.ris_num_elements)

        self.queue_len = random.randint(0, 3)
        self.hol_delay = 0
        self.last_channel = 0
        self.last_ris_mode = 0
        self.last_tx_result = "listen"
        self.last_collision = 0

        self.collision_history = deque(maxlen=config.history_len)
        self.delay_history = deque(maxlen=config.history_len)

        for _ in range(config.history_len):
            self.collision_history.append(0)
            self.delay_history.append(0)

    def packet_arrival(self):
        if random.random() < self.cfg.queue_arrival_prob:
            if self.queue_len < self.cfg.max_queue_len:
                self.queue_len += 1

    def update_delay(self):
        if self.queue_len > 0:
            self.hol_delay += 1
        else:
            self.hol_delay = 0

    def get_mac_obs(self):
        return {
            "queue_len": self.queue_len,
            "hol_delay": self.hol_delay,
            "last_channel": self.last_channel,
            "last_ris_mode": self.last_ris_mode,
            "last_tx_result": self.last_tx_result,
            "avg_collision": np.mean(self.collision_history),
            "avg_delay": np.mean(self.delay_history),
        }

    def get_ris_obs(self):
        return {
            "queue_len": self.queue_len,
            "hol_delay": self.hol_delay,
            "last_channel": self.last_channel,
            "last_ris_mode": self.last_ris_mode,
            "last_tx_result": self.last_tx_result,
        }

    def select_actions(self):
        mac_obs = self.get_mac_obs()
        ris_obs = self.get_ris_obs()

        mac_action = self.mac_agent.select_action(mac_obs)
        ris_action = self.ris_agent.select_action(ris_obs)

        return mac_obs, ris_obs, mac_action, ris_action

    def on_success(self, channel, ris_mode):
        self.last_channel = channel
        self.last_ris_mode = ris_mode
        self.last_tx_result = "success"
        self.last_collision = 0

        if self.queue_len > 0:
            self.queue_len -= 1

        if self.queue_len == 0:
            self.hol_delay = 0

        self.collision_history.append(0)
        self.delay_history.append(self.hol_delay)

    def on_mac_contention_loss(self, channel, ris_mode):
        self.last_channel = channel
        self.last_ris_mode = ris_mode
        self.last_tx_result = "mac_contention_loss"
        self.last_collision = 1
        self.collision_history.append(1)
        self.delay_history.append(self.hol_delay)

    def on_ris_conflict_loss(self, channel, ris_mode):
        self.last_channel = channel
        self.last_ris_mode = ris_mode
        self.last_tx_result = "ris_conflict_loss"
        self.last_collision = 0
        self.collision_history.append(0)
        self.delay_history.append(self.hol_delay)

    def on_tx_fail(self, channel, ris_mode):
        self.last_channel = channel
        self.last_ris_mode = ris_mode
        self.last_tx_result = "tx_fail"
        self.last_collision = 0
        self.collision_history.append(0)
        self.delay_history.append(self.hol_delay)

    def on_listen(self):
        self.last_channel = 0
        self.last_tx_result = "listen"
        self.last_collision = 0
        self.collision_history.append(0)
        self.delay_history.append(self.hol_delay)
