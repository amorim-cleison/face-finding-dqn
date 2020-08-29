text_width = 105


def log_start(episode):
    print("=" * text_width)
    print(f"EPISODE {episode}")
    print("-" * text_width)
    print(
        f"{'Step':4} | {'State':<35} {'Action':<18} {'Reward':<12} {'Accuracy':<12} {'Loss':<12}"
    )
    print("-" * text_width)


def log_progress(step, state, action, action_name, reward, loss, accuracy,
                 **kwargs):
    _action = f"{action} ({action_name})"
    _reward = f"{reward:.6f}"
    _loss = f"{loss:.6f}"
    _accuracy = f"{accuracy:.6f}"
    print(
        f"{step:<4} | {state:<35} {_action:<18} {_reward:<12} {_accuracy:<12} {_loss:<12}"
    )


def log_finish(success, steps):
    print("-" * text_width)
    if success:
        print(f" -> Completed in {steps} step(s)")
    else:
        print(" -> Failed to complete trial")
    print()


def log_model(model):
    print()
    print("-" * text_width)
    print(model.summary())
    print("-" * text_width)
    print()


def log_args(args):
    print("=" * text_width)
    print("ARGUMENTS")
    print("-" * text_width)
    print(args)
    print("-" * text_width)
    print()
