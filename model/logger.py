text_width = 95


def log_start(episode):
    print("=" * text_width)
    print(f"EPISODE {episode}")
    print("-" * text_width)
    print(
        f"{'Step':5} | {'State':<32}\t {'Action':<12}\t {'Reward':<10}\t {'Loss'}"
    )
    print("-" * text_width)


def log_progress(step, state, action, action_name, reward, loss, **kwargs):
    _action = f"{action} ({action_name})"
    print(
        f"{step:<5} | {state:<32}\t {_action:<12}\t {reward:.6f}\t {loss:.6f}")


def log_finish(success, steps):
    print("-" * text_width)
    if success:
        print(f" -> Completed in {steps} step(s)")
    else:
        print(" -> Failed to complete trial")
    print()
