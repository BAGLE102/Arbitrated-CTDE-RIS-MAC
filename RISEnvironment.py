class RISEnvironment:
    def __init__(self, config: Config):
        self.cfg = config
        self.ues = [UEAgent(k, config) for k in range(config.num_ues)]
        self.ap = APArbitrator(config)

        self.reset_statistics()

    def reset_statistics(self):
        self.total_success = 0
        self.total_mac_contention = 0
        self.total_ris_conflict = 0
        self.total_tx_fail = 0
        self.total_listen = 0

    def reset(self):
        self.ues = [UEAgent(k, self.cfg) for k in range(self.cfg.num_ues)]
        self.ap = APArbitrator(self.cfg)
        self.reset_statistics()

    def estimate_channel_gain(self, ue_id, theta_star):
        """
        這裡先用簡單抽象值代替實際通道模型
        """
        base_gain = np.random.uniform(0.5, 2.0)
        ris_bonus = np.random.uniform(0.0, 1.0)
        return base_gain + 0.3 * ris_bonus + 0.1 * theta_star

    def compute_sinr(self, gain):
        noise = np.random.uniform(0.2, 1.0)
        sinr = gain / noise
        return sinr

    def minislot1_collect_requests(self):
        requests = []
        mac_obs_map = {}
        ris_obs_map = {}

        for ue in self.ues:
            ue.packet_arrival()
            ue.update_delay()

            mac_obs, ris_obs, mac_action, ris_action = ue.select_actions()

            mac_obs_map[ue.id] = mac_obs
            ris_obs_map[ue.id] = ris_obs

            req = {
                "ue_id": ue.id,
                "mac_action": mac_action,   # 0=Listen, 1~N=channel
                "ris_action": ris_action,
                "queue_len": ue.queue_len,
                "hol_delay": ue.hol_delay,
                "avg_collision": mac_obs["avg_collision"],
                "estimated_gain": np.random.uniform(0.5, 2.0),  # 初步估計
            }
            requests.append(req)

        return requests, mac_obs_map, ris_obs_map

    def minislot2_arbitration(self, requests):
        theta_star, ris_conflict_users, ris_conflict_flag = self.ap.arbitrate_ris(requests)
        mac_results = self.ap.arbitrate_mac(requests, ris_conflict_users)
        return theta_star, ris_conflict_users, ris_conflict_flag, mac_results

    def minislot3_execute_transmission(self, theta_star, requests, mac_results):
        final_results = {}

        for req in requests:
            ue_id = req["ue_id"]
            mac_action = req["mac_action"]

            if mac_action == 0:
                final_results[ue_id] = {
                    "status": "listen",
                    "channel": 0,
                    "sinr": None,
                }
                continue

            if ue_id not in mac_results:
                final_results[ue_id] = {
                    "status": "tx_fail",
                    "channel": mac_action,
                    "sinr": None,
                }
                continue

            status = mac_results[ue_id]["status"]
            channel = mac_results[ue_id]["channel"]

            if status == "ris_conflict_loss":
                final_results[ue_id] = {
                    "status": "ris_conflict_loss",
                    "channel": channel,
                    "sinr": None,
                }

            elif status == "mac_contention_loss":
                final_results[ue_id] = {
                    "status": "mac_contention_loss",
                    "channel": channel,
                    "sinr": None,
                }

            elif status == "granted":
                gain = self.estimate_channel_gain(ue_id, theta_star)
                sinr = self.compute_sinr(gain)

                if sinr >= self.cfg.sinr_threshold:
                    final_results[ue_id] = {
                        "status": "success",
                        "channel": channel,
                        "sinr": sinr,
                    }
                else:
                    final_results[ue_id] = {
                        "status": "tx_fail",
                        "channel": channel,
                        "sinr": sinr,
                    }

        return final_results

    def compute_rewards(self, final_results):
        rewards = {}
        success_count = sum(1 for r in final_results.values() if r["status"] == "success")
        tx_count = sum(1 for r in final_results.values() if r["status"] != "listen")
        mac_contention_count = sum(1 for r in final_results.values() if r["status"] == "mac_contention_loss")
        ris_conflict_count = sum(1 for r in final_results.values() if r["status"] == "ris_conflict_loss")

        throughput_reward = success_count
        mac_penalty = mac_contention_count / max(1, tx_count)
        ris_penalty = ris_conflict_count / max(1, tx_count)

        system_reward = throughput_reward - mac_penalty - ris_penalty

        for ue_id, result in final_results.items():
            status = result["status"]

            if status == "success":
                local_reward = self.cfg.r_success
            elif status == "mac_contention_loss":
                local_reward = self.cfg.r_mac_contention
            elif status == "ris_conflict_loss":
                local_reward = self.cfg.r_ris_conflict
            elif status == "tx_fail":
                local_reward = self.cfg.r_tx_fail
            else:
                local_reward = self.cfg.r_listen

            total_reward = 0.7 * local_reward + 0.3 * system_reward
            rewards[ue_id] = total_reward

        return rewards

    def update_ue_states(self, theta_star, final_results):
        for ue in self.ues:
            result = final_results[ue.id]
            status = result["status"]
            channel = result["channel"]

            if status == "success":
                ue.on_success(channel, theta_star)
                self.total_success += 1

            elif status == "mac_contention_loss":
                ue.on_mac_contention_loss(channel, theta_star)
                self.total_mac_contention += 1

            elif status == "ris_conflict_loss":
                ue.on_ris_conflict_loss(channel, theta_star)
                self.total_ris_conflict += 1

            elif status == "tx_fail":
                ue.on_tx_fail(channel, theta_star)
                self.total_tx_fail += 1

            else:
                ue.on_listen()
                self.total_listen += 1

    def step(self):
        # minislot 1
        requests, mac_obs_map, ris_obs_map = self.minislot1_collect_requests()

        # minislot 2
        theta_star, ris_conflict_users, ris_conflict_flag, mac_results = self.minislot2_arbitration(requests)

        # minislot 3
        final_results = self.minislot3_execute_transmission(theta_star, requests, mac_results)

        # minislot 4
        rewards = self.compute_rewards(final_results)
        self.update_ue_states(theta_star, final_results)

        return {
            "theta_star": theta_star,
            "requests": requests,
            "final_results": final_results,
            "rewards": rewards,
            "ris_conflict_users": ris_conflict_users,
            "ris_conflict_flag": ris_conflict_flag,
        }

    def print_statistics(self):
        print("=== Simulation Statistics ===")
        print(f"Success             : {self.total_success}")
        print(f"MAC contention loss : {self.total_mac_contention}")
        print(f"RIS conflict loss   : {self.total_ris_conflict}")
        print(f"TX fail             : {self.total_tx_fail}")
        print(f"Listen              : {self.total_listen}")
