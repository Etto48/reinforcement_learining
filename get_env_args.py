def get_env_args(env_name: str):
    env_args = {
        "LunarLander-v3": {"continuous": True}
    }
    ret = env_args.get(env_name, {})
    ret["render_mode"] = "rgb_array"
    return ret