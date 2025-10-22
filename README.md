# Main Development Space for NRL - MCTF
This the main repository where the bulk of team MCTF 3 will develop their AI training agents and testing.

Competition 2025 Website: https://mctf2025.com/installation

Original Pyquaticus Repository: https://github.com/mit-ll-trusted-autonomy/pyquaticus/tree/main

# Performance

This project utilizes RLib extensively. Performance is greatly improved through the use of a GPU. However, right after the GPU, the # of CPU cores is the
next largest bottleneck. The current configuration for testing AI agent training is optimized for a GTX 1070 GPU and a 4-core Intel CPU. Change the following
segment of code in pyquaticus/rl_test/train_3v3.py and pyquaticus/rl_test/train_2v2.py. If you have a GPU with more than 2GB of VRAM and a CPU with more than
4-cores, please try to run and optimize the configuration. Current best per loop (out of 8000) time is ~16 secs for train_3v3.py. Even a few seconds saved can
reduce AI training time by hours.

```
PPOConfig()
    .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)  # Disable alpha API stack (legacy RLlib),  #Not using the Alpha Rllib (api_stack False) 
    .environment(env='pyquaticus') # Environment setup
    .env_runners(num_env_runners=4, num_cpus_per_env_runner=1) # Set runners (ideal: num_env_runners = #CPU cores) and resource allocation
    .resources(num_gpus=NUM_GPUS) #Enable GPU usage
    .framework("torch") # Ensure we use PyTorch backend (most common but TensorFlow works too)
    #.learners(use_fp16=True) #NVIDIA GPU 1070 GTX Specific. Remove if needed.
    #.reporting(min_sample_timesteps_per_iteration=2048) # Rollout / logging frequency
)
    ppo_config.update_from_dict({ #Optimized training parameters specific to GTX 1070. May need changing if not using same GPU. If unsure leave as-is.
    "train_batch_size": 2048,
    "sgd_minibatch_size": 256,
    "num_sgd_iter": 8,
    "lr": 3e-4,
    "gamma": 0.99,
    "lambda": 0.95,
    "clip_param": 0.2,
    "grad_clip": 0.5,
    "vf_clip_param": 10.0,
})
```

