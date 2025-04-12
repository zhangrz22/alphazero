from .node import MCTSNode
from .config import MCTSConfig
from env.base_env import BaseGame

import numpy as np
from other_algo.heuristic import go_heuristic_evaluation

class UCTMCTSConfig(MCTSConfig):
    def __init__(
        self,
        n_rollout:int = 1,
        rollout_depth: int = 25,
        *args, **kwargs
    ):
        MCTSConfig.__init__(self, *args, **kwargs)
        self.n_rollout = n_rollout
        self.rollout_depth = rollout_depth 


class UCTMCTS:
    def __init__(self, init_env:BaseGame, config: UCTMCTSConfig, root:MCTSNode=None):
        self.config = config
        self.root = root
        if root is None:
            self.init_tree(init_env)
        self.root.cut_parent()
    
    def init_tree(self, init_env:BaseGame):
        # initialize the tree with the current state
        # fork the environment to avoid side effects
        env = init_env.fork()
        self.root = MCTSNode(
            action=None, env=env, reward=0,
        )
    
    def get_subtree(self, action:int):
        # return a subtree with root as the child of the current root
        # the subtree represents the state after taking action
        if self.root.has_child(action):
            new_root = self.root.get_child(action)
            return UCTMCTS(new_root.env, self.config, new_root)
        else:
            return None
    
    def uct_action_select(self, node:MCTSNode) -> int:
        # select the best action based on UCB when expanding the tree       
        legal_actions = np.where(node.action_mask == 1)[0]
        
        total_visits = node.child_N_visit.sum()
        bestValue = float("-inf")
        bestNodes = []
        for a in legal_actions:
            n_i = node.child_N_visit[a]
            q_i = node.child_V_total[a]
            uct = q_i/n_i + self.config.C * np.sqrt(np.log(total_visits) / n_i)
            if uct > bestValue:
                bestValue = uct
                bestNodes = [a]
            elif uct == bestValue:
                bestNodes.append(a)
        return np.random.choice(bestNodes)
    
    def backup(self, node:MCTSNode, value:float) -> None:
        # backup the value of the leaf node to the root
        # update N_visit and V_total of each node in the path
        current_node = node
        current_value = value
        while current_node.parent is not None:
            action = current_node.action
            parent = current_node.parent
            parent.child_N_visit[action] += 1
            parent.child_V_total[action] += current_value
            current_value = -current_value
            current_node = parent 
            
    def rollout(self, node:MCTSNode) -> float:
        # simulate the game until the end
        # return the reward of the game
        # NOTE: the reward should be convert to the perspective of the current player!

        env = node.env.fork()
        current_player = env.current_player
        score = 0
        depth = 0

        # 随机模拟直到游戏结束
        while not env.ended and depth < self.config.rollout_depth:
            valid_actions = np.where(env.action_mask)[0]
            if len(valid_actions) == 0:
                break
            action = np.random.choice(valid_actions)
            # 捕获 step 返回值
            _, reward, done = env.step(action)
            depth += 1
            if done:
                score = reward

        if env.ended:
            # 调整奖励视角：若模拟结束时当前玩家与节点中玩家不一致，则翻转奖励
            if env.current_player != current_player:
                score *= -1
            return score
        else:
            score = go_heuristic_evaluation(env)
            # 如果模拟结束时的当前玩家和初始玩家不一致，则翻转评估得分
            if env.current_player != current_player:
                score *= -1
            return score
    
    def pick_leaf(self) -> MCTSNode:
        # select the leaf node to expand
        # the leaf node is the node that has not been expanded
        # create and return a new node if game is not ended        
        node = self.root
        while not node.done:
            legal_actions = np.where(node.action_mask == 1)[0]
            unvisited = [a for a in legal_actions if not node.has_child(a)]
            if unvisited:
                action = np.random.choice(unvisited)
                node = node.add_child(action)
                return node
            else:
                action = self.uct_action_select(node)
                node = node.get_child(action)
        return node
    

    def get_policy(self, node:MCTSNode = None) -> np.ndarray:
        # return the policy of the tree(root) after the search
        # the policy comes from the visit count of each action         
        if node is None:
            node = self.root
        policy = np.zeros(node.n_action, dtype=np.float32)
        legal_actions = np.where(node.action_mask == 1)[0]
        total_visits = node.child_N_visit[legal_actions].sum()
        if total_visits == 0:
            policy[legal_actions] = 1.0 / len(legal_actions)
        else:
            policy[legal_actions] = node.child_N_visit[legal_actions] / total_visits
        return policy

    def search(self):
        # search the tree for n_search times
        # eachtime, pick a leaf node, rollout the game (if game is not ended) 
        #   for n_rollout times, and backup the value.
        # return the policy of the tree after the search
        for _ in range(self.config.n_search):
            # print("search")
            leaf = self.pick_leaf()
            value = 0
            if leaf.done:
                value = leaf.reward
            else:
                for _ in range(self.config.n_rollout):
                    # print("rollout")
                    value += self.rollout(leaf)
                value /= self.config.n_rollout
            self.backup(leaf, value)
        return self.get_policy(self.root)