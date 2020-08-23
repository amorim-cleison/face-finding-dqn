def log_start(episode):
    print("=" * 90)
    print(f"EPISODE {episode}")
    print("-" * 90)
    print(f"{'Step':5} | {'State':<40}\t {'Action'}\t {'Reward':<10}\t {'Loss'}")
    print("-" * 90)


def log_progress(step, state, action, reward, loss, **kwargs):
    print(
        f"{step:<5} | S: {state:<40}\t A: {action}\t R: {reward:.2f}\t {loss:.6f}"
    )


def log_finish(success, steps):
    print("-" * 90)
    if success:
        print(f" -> Completed in {steps} step(s)")
    else:
        print(" -> Failed to complete trial")
    print()
