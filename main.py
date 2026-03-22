def run_simulation():
    cfg = Config()
    env = RISEnvironment(cfg)

    for ep in range(cfg.num_episodes):
        env.reset()
        print(f"\n========== Episode {ep+1} ==========")

        for t in range(cfg.episode_length):
            info = env.step()

            if t < 5:  # 前幾個 slot 印出來看
                print(f"\n[Slot {t+1}] theta_star = {info['theta_star']}")
                for ue_id, result in info["final_results"].items():
                    print(f"UE {ue_id}: {result}")

        env.print_statistics()


if __name__ == "__main__":
    run_simulation()
