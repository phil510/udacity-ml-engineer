import abc
import numpy as np
import os
import sys

def train(agent, env = None, train_steps = 10000, 
          hooks = [], vector_env = False):
    assert (agent.training), 'TODO'
    
    if env is None:
        env = agent.env
    
    for hook in hooks:
        hook.begin(env, agent)
        
    obs = env.reset()
    terminal = False
    for hook in hooks:
        hook.reset(env, agent)
    
    for _ in range(train_steps):
        action = agent.action(obs)
        next_obs, reward, terminal, _ = env.step(action)
        agent.update(obs, action, reward, next_obs, terminal)
        
        for hook in hooks:
            hook.step(env, agent, obs, action, reward, next_obs, terminal)
        
        obs = next_obs
        
        if vector_env:
            terminal = terminal[0]
    
        if terminal:
            if not vector_env:
                env.reset()
            terminal = False
            for hook in hooks:
                hook.reset(env, agent)
    
    for hook in hooks:
        hook.end(env, agent)
        
def evaluate(agent, env = None, eval_steps = 1000,
             hooks = [], vector_env = False):
    assert (not agent.training), 'TODO'
    
    if env is None:
        env = agent.env
    
    for hook in hooks:
        hook.begin(env, agent)
        
    obs = env.reset()
    terminal = False
    for hook in hooks:
        hook.reset(env, agent)
    
    for _ in range(eval_steps):
        action = agent.action(obs)
        next_obs, reward, terminal, _ = env.step(action)
        agent.update(obs, action, reward, next_obs, terminal)
        
        for hook in hooks:
            hook.step(env, agent, obs, action, reward, next_obs, terminal)
        
        obs = next_obs
        
        if vector_env:
            terminal = terminal[0]
    
        if terminal:
            env.reset()
            terminal = False
            for hook in hooks:
                hook.reset(env, agent)
    
    for hook in hooks:
        hook.end(env, agent)        
        
def train_and_evaluate(agent, env = None, 
                       train_steps = 10000, 
                       eval_steps = 1000, 
                       eval_every = 1000,
                       train_hooks = [], 
                       eval_hooks = [], 
                       vector_env = False):
    if env is None:
        use_agent_env = True
    else:
        use_agent_env = False
    
    intervals = np.ceil(train_steps / eval_every).astype(int)
    steps_taken = 0
    reset_at_start = True
    
    for i in range(intervals):
        if i == (intervals - 1):
            steps = (train_steps % eval_every)
            if steps == 0:
                steps = eval_every
        else:
            steps = eval_every
        
        # training phase
        agent.train()
        print('Beginning training')
        if use_agent_env:
            env = agent.env
        
        if i == 0:
            for hook in train_hooks:
                hook.begin(env, agent)
        
        obs = env.reset()
        terminal = False
        if reset_at_start: # to avoid resetting the hooks twice in a row
            for hook in train_hooks:
                hook.reset(env, agent)
        
        for j in range(steps):
            action = agent.action(obs)
            next_obs, reward, terminal, _ = env.step(action)
            agent.update(obs, action, reward, next_obs, terminal)
            
            for hook in train_hooks:
                hook.step(env, agent, obs, action, reward, next_obs, terminal)
            
            obs = next_obs
            reset_at_start = True
            
            if vector_env:
                terminal = terminal[0]
        
            if terminal:
                if not vector_env:
                    env.reset()
                terminal = False
                for hook in train_hooks:
                    hook.reset(env, agent)
                reset_at_start = False
        
        if i == (intervals - 1):
            for hook in train_hooks:
                hook.end(env, agent)
                
        # evaluation phase
        agent.eval()
        print('Beginning evaluation')
        if use_agent_env:
            env = agent.env
        
        evaluate(agent, env, eval_steps = eval_steps,
                 hooks = eval_hooks, vector_env = vector_env)
        
        steps_taken += steps
        if i == (intervals - 1):
            assert (steps_taken == train_steps), 'TODO'