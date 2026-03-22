class APArbitrator:
    def __init__(self, config: Config):
        self.cfg = config
        self.last_theta = 0

        # 建立 RIS codebook
        self.ris_codebook = self._build_ris_codebook()

    def _build_ris_codebook(self):
        codebook = []
        for _ in range(self.cfg.ris_codebook_size):
            mode = np.random.uniform(-1.0, 1.0, size=self.cfg.ris_num_elements)
            codebook.append(mode)
        return codebook

    def map_continuous_to_mode(self, ris_vector):
        """
        找最近的 codebook mode index
        """
        min_dist = float("inf")
        best_mode = 0

        for idx, mode_vec in enumerate(self.ris_codebook):
            dist = np.linalg.norm(ris_vector - mode_vec)
            if dist < min_dist:
                min_dist = dist
                best_mode = idx

        return best_mode

    def arbitrate_ris(self, requests):
        """
        RIS arbitration:
        - 若沒人 transmit -> 維持上次模式
        - 若只有一個 transmitting UE -> 用該 mode
        - 若多人 transmitting:
            * 若要求 mode 一樣 -> coordinated success
            * 若不同 -> RIS conflict
        """
        tx_requests = [req for req in requests if req["mac_action"] != 0]

        if len(tx_requests) == 0:
            return self.last_theta, [], False

        requested_modes = {}
        for req in tx_requests:
            requested_modes[req["ue_id"]] = self.map_continuous_to_mode(req["ris_action"])

        unique_modes = set(requested_modes.values())

        if len(unique_modes) == 1:
            theta_star = list(unique_modes)[0]
            self.last_theta = theta_star
            return theta_star, [], False
        else:
            # 多人要求不同 mode -> RIS conflict
            conflict_users = list(requested_modes.keys())
            theta_star = self.last_theta
            return theta_star, conflict_users, True

    def compute_priority_score(self, req):
        """
        score 越高越有機會被 grant
        """
        gain = req["estimated_gain"]
        queue_len = req["queue_len"]
        delay = req["hol_delay"]
        collision_hist = req["avg_collision"]

        score = (
            self.cfg.w_gain * gain
            + self.cfg.w_queue * queue_len
            + self.cfg.w_delay * delay
            - self.cfg.w_collision_hist * collision_hist
        )
        return score

    def arbitrate_mac(self, requests, ris_conflict_users):
        """
        在 minislot 2 做 channel-level arbitration
        """
        results = {}
        tx_requests = [req for req in requests if req["mac_action"] != 0]

        # 若 RIS conflict，先標記這些 transmitting UE
        if len(ris_conflict_users) > 0:
            for req in tx_requests:
                if req["ue_id"] in ris_conflict_users:
                    results[req["ue_id"]] = {
                        "status": "ris_conflict_loss",
                        "channel": req["mac_action"],
                    }
            return results

        channel_groups = defaultdict(list)
        for req in tx_requests:
            ch = req["mac_action"]
            channel_groups[ch].append(req)

        for ch, group in channel_groups.items():
            if len(group) == 1:
                ue_id = group[0]["ue_id"]
                results[ue_id] = {
                    "status": "granted",
                    "channel": ch,
                }
            else:
                # 同頻競爭 -> 交由 AP 根據資訊仲裁
                scored_group = []
                for req in group:
                    score = self.compute_priority_score(req)
                    scored_group.append((score, req))

                scored_group.sort(key=lambda x: x[0], reverse=True)

                winner_req = scored_group[0][1]
                winner_id = winner_req["ue_id"]

                results[winner_id] = {
                    "status": "granted",
                    "channel": ch,
                }

                for _, loser_req in scored_group[1:]:
                    loser_id = loser_req["ue_id"]
                    results[loser_id] = {
                        "status": "mac_contention_loss",
                        "channel": ch,
                    }

        return results
