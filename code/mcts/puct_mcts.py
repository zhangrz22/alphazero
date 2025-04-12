from .node import MCTSNode, INF
from .config import MCTSConfig
from env.base_env import BaseGame

from model.linear_model_trainer import NumpyLinearModelTrainer
import numpy as np


class PUCTMCTS:
    def __init__(self, init_env:BaseGame, model: NumpyLinearModelTrainer, config: MCTSConfig, root:MCTSNode=None):
        self.model = model
        self.config = config
        self.root = root
        if root is None:
            self.init_tree(init_env)
        self.root.cut_parent()
    
    def init_tree(self, init_env:BaseGame):
        env = init_env.fork()
        obs = env.observation
        self.root = MCTSNode(
            action=None, env=env, reward=0
        )
        # compute and save predicted policy
        child_prior, _ = self.model.predict(env.compute_canonical_form_obs(obs, env.current_player))
        self.root.set_prior(child_prior)
    
    def get_subtree(self, action:int):
        # return a subtree with root as the child of the current root
        # the subtree represents the state after taking action
        if self.root.has_child(action):
            new_root = self.root.get_child(action)
            return PUCTMCTS(new_root.env, self.model, self.config, new_root)
        else:
            return None
    
    def puct_action_select(self, node:MCTSNode):
        # select the best action based on PUCB when expanding the tree
        legal_actions = np.where(node.action_mask == 1)[0]
        
        total_visits = node.child_N_visit.sum()
        bestValue = float("-inf")
        bestNodes = []
        
        for a in legal_actions:
            if node.child_N_visit[a] == 0:
                return a
            
            # PUCB公式：Q(s,a) + C * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
            q_value = node.child_V_total[a] / node.child_N_visit[a]
            prior_score = self.config.C * node.child_priors[a] * np.sqrt(total_visits) / (1 + node.child_N_visit[a])
            puct = q_value + prior_score
            
            if puct > bestValue:
                bestValue = puct
                bestNodes = [a]
            elif puct == bestValue:
                bestNodes.append(a)
                
        return np.random.choice(bestNodes)
    
    def backup(self, node:MCTSNode, value):
        # backup the value of the leaf node to the root
        # update N_visit and V_total of each node in the path
        current_node = node
        current_value = value
        
        while current_node.parent is not None:
            action = current_node.action
            parent = current_node.parent
            parent.child_N_visit[action] += 1
            parent.child_V_total[action] += current_value
            current_value = -current_value  # 翻转价值，因为是零和游戏
            current_node = parent
    
    def pick_leaf(self):
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
                # 对新节点设置先验概率
                child_prior, _ = self.model.predict(node.env.compute_canonical_form_obs(node.env.observation, node.env.current_player))
                node.set_prior(child_prior)
                return node
            else:
                action = self.puct_action_select(node)
                node = node.get_child(action)
        return node
    
    def get_policy(self, node:MCTSNode = None):
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
        for _ in range(self.config.n_search):
            leaf = self.pick_leaf()
            value = 0
            if leaf.done:
                value = leaf.reward
            else:
                # 使用价值网络预测节点价值
                _, value = self.model.predict(leaf.env.compute_canonical_form_obs(leaf.env.observation, leaf.env.current_player))
            self.backup(leaf, value)
            
        return self.get_policy(self.root)