# Reinforcement Learning assignment
MiniHack RL models for Quest-Hard-v0

# Environment
### Minihack [[repo link](https://github.com/facebookresearch/minihack)]


Goal is to create an agent able to beat the environment "Quest-Hard-v0" within minihack


![Screenshot (17)](https://user-images.githubusercontent.com/42907395/141273959-517c8af9-c9b1-4ded-a62f-343b9a631958.png)


Multiple tasks a required in order to beat this environment.

- Exit the maze
- Cross the lava river
- Defeat the boss monster
- Exit the level at the stairs down


# Create Conda Env from environment.yml

    conda env create -f environment.yml -n assignment
    
# Models
### DQN
Training:     ```python3 dqn.py```

Testing:      ```python3 dqn.py -test <path_to_model>```

Random Agent: ```python3 dqn.py -random```



### REINFORCE
Training:     ```python3 reinforce.py```
