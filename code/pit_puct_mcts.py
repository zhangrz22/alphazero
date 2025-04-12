from env import *
from players import *
from mcts.puct_mcts import MCTSConfig as PUCTMCTSConfig
from tqdm import trange, tqdm
from multiprocessing import Process
from torch.distributed.elastic.multiprocessing import Std, start_processes

import logging
logger = logging.getLogger(__name__)

import numpy as np

def log_devide_line(n=50):
    logger.info("--"*n)

def pit(game:BaseGame, player1:PUCTPlayer, player2:PUCTPlayer, log_output:bool=False):
    game.reset()
    if log_output:
        logger.info(f"start playing {type(game)}")
        log_devide_line()
    reward = 0
    
    for player in [player1, player2]:
        if hasattr(player, "clear"):
            player.clear()
            
    while True:
        a1 = player1.play(game)
        _, reward, done = game.step(a1)
        if hasattr(player2, "opp_play"):
            player2.opp_play(a1)
        if log_output:
            logger.info(f"Player 1 ({player1}) move: {a1}")
            logger.info(game.to_string())
            log_devide_line()
        if done:
            break
        a2 = player2.play(game)
        _, reward, done = game.step(a2)
        if hasattr(player1, "opp_play"):
            player1.opp_play(a2)
        if log_output:
            logger.info(f"Player 2 ({player2}) move: {a2}")
            logger.info(game.to_string())
            log_devide_line()
        if done:
            reward *= -1
            break
    if log_output:
        if reward == 1:
            logger.info(f"Player 1 ({player1}) win")
        elif reward == -1:
            logger.info(f"Player 2 ({player2}) win")
        else:
            logger.info("Draw")
    # print(game.observation, reward)
    return reward

def multi_match(game:BaseGame, player1:PUCTPlayer, player2:PUCTPlayer, n_match=100, disable_tqdm=False):
    assert n_match % 2 == 0 and n_match > 1, "n_match should be an even number greater than 1"
    n_p1_win, n_p2_win, n_draw = 0, 0, 0
    for _ in trange(n_match//2, disable=disable_tqdm):
        reward = pit(game, player1, player2, log_output=False)
        if reward == 1:
            n_p1_win += 1
        elif reward == -1:
            n_p2_win += 1
        else:
            n_draw += 1
    first_play = n_p1_win, n_p2_win, n_draw
    for _ in trange(n_match//2, disable=disable_tqdm):
        reward = pit(game, player2, player1, log_output=False)
        if reward == 1:
            n_p2_win += 1
        elif reward == -1:
            n_p1_win += 1
        else:
            n_draw += 1
    latter_play = n_p1_win - first_play[0], n_p2_win - first_play[1], n_draw - first_play[2]
    return (n_p1_win, n_p2_win, n_draw), first_play, latter_play
