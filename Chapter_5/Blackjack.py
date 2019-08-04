#==================================================================
# Python3
# Copyright
# 2019 Ye Xiang (xiang_ye@outlook.com)
#==================================================================

import copy
import numpy as np
from enum import Enum
import matplotlib.pyplot as mplt_pyplt
from mpl_toolkits.mplot3d import Axes3D
from progressbar import ProgressBar
import seaborn as sbn

ACTION_STICK = 0
ACTION_HIT   = 1

class PlayerState:
    def __init__(self):
        self.card_sum = 0
        self.used_ace_num = 0
        self.dealers_showing = 0
    def reset(self):
        self.card_sum = 0
        self.used_ace_num = 0
        self.dealers_showing = 0

    def __eq__(self,rhs):
        if self.card_sum == rhs.card_sum \
            and self.dealers_showing == rhs.dealers_showing \
            and self.used_ace_num == rhs.used_ace_num:
            return True
        else:
            return False

class TrajectoryElement:
    def __init__(self, player_state = PlayerState(), action = ACTION_STICK, reward = 0.0):
        self.state = player_state
        self.action = action
        self.reward = reward

class Player:
    def __init__(self):
        self._trajectory = []
        #[sum, dealer's showing, used_ace_num]
        self._state = PlayerState() 

    def _get_card(self):
        card = np.random.randint(1, 14)
        card = min(card, 10)
        return card

    def _get_action(self):
        return ACTION_HIT if self._state.card_sum < 20 else ACTION_STICK
    
    def start_game(self, dealers_showing_card):
        self._state.reset()
        self._state.dealers_showing = dealers_showing_card
        self._trajectory = []
        while self._state.card_sum < 12:
            card = self._get_card()
            self._update_state(card)
    
    def _update_state(self, card):
        if card == 1:
            self._state.card_sum += 11
            self._state.used_ace_num += 1
        else:
            self._state.card_sum += card
        while self._state.card_sum > 21 and self._state.used_ace_num > 0:
            self._state.card_sum -= 10
            self._state.used_ace_num -= 1     
        if self._state.card_sum >=12 and self._state.card_sum <= 21:
            self._trajectory.append(TrajectoryElement(player_state = copy.copy(self._state), \
                reward = 0, action = self._get_action()))
    
    def play_game(self):
        while self._get_action() == ACTION_HIT and self._state.card_sum <= 21:
            card = self._get_card()
            self._update_state(card)

    def set_game_result(self, final_reward):
        self._trajectory[-1].reward = final_reward

    @property
    def sum(self):
        return self._state.card_sum

    @property
    def trajectory(self):
        return self._trajectory
    
    

class Dealer(Player):
    def _get_action(self):
        return ACTION_HIT if self._state.card_sum < 17 else ACTION_STICK

    def start_game(self):
        self._state.card_sum = 0
        self._state.used_ace_num = 0
        card1 = self._get_card()
        card2 = self._get_card()
        self._update_state(card1)
        self._update_state(card2)  
        #return showing card randomly
        if np.random.choice([0, 1]) == 0:
            return card1
        else:
            return card2

    def _update_state(self, card):
        if card == 1:
            self._state.card_sum += 11
            self._state.used_ace_num += 1
        else:
            self._state.card_sum += card
        while self._state.card_sum > 21 and self._state.used_ace_num > 0:
            self._state.card_sum -= 10
            self._state.used_ace_num -= 1    

class Player_MC_ES(Player):
        
    def __init__(self):
        Player.__init__(self)
        self._policy = np.ones((10, 10, 2), dtype = np.int)
        self._policy[(20-12):(22-12), :, :] = ACTION_STICK
        self.__initial_state = False
        self.__initial_action = None

    def _get_action(self):
        if self.__initial_state:
            if self.__initial_action == None:
                self.__initial_action = np.random.choice([ACTION_STICK, ACTION_HIT])
            return self.__initial_action
        if self._state.card_sum <= 21:
            return self._policy[self._state.card_sum - 12][self._state.dealers_showing - 1][int(self._state.used_ace_num > 0)]
        else:
            return ACTION_STICK
    
    def start_game(self, dealers_showing_card):
        self.__initial_state = True
        self.__initial_action = None
        self._trajectory = []
        self._state.reset()
        self._state.dealers_showing = dealers_showing_card
        while self._state.card_sum < 12:
            card = self._get_card()
            self._update_state(card)
        self._trajectory.append(TrajectoryElement(player_state = copy.copy(self._state), reward = 0, \
            action = self._get_action()))
    
    
    def play_game(self):
        while self._get_action() == ACTION_HIT and self._state.card_sum <= 21:
            self.__initial_state = False
            self.__initial_action = None
            card = self._get_card()
            self._update_state(card)
    

    def set_policy(self, state, val):
        self._policy[state.card_sum - 12,state.dealers_showing - 1][int(state.used_ace_num > 0)] = val

    @property
    def policy(self):
        return self._policy

class Player_target(Player):
    def start_game(self):
        self._state.reset()
        self._state.dealers_showing = 2
        self._state.card_sum = 13
        self._state.used_ace_num = 1
        self._trajectory = []
        self._trajectory.append(TrajectoryElement(player_state = self._state, \
            action = self._get_action(), reward = 0))

    def get_action(self, state):
        return ACTION_HIT if state.card_sum < 20 else ACTION_STICK

class Player_behavior(Player_target):
    def _get_action(self):
        return np.random.choice([ACTION_STICK, ACTION_HIT])

class Blackjack:
    def __init__(self):
        self.__history_Vs = []
        #[card_sum, dealer's showing, usable ace]
        self.__Vs = np.zeros((10, 10, 2))
        self.__Vs_count = np.zeros((10,10,2), dtype = np.int)

        #[card_sum, dealer's showing, usable ace, action]
        self.__Q = np.zeros((10, 10, 2, 2))
        self.__Q_count = np.zeros((10, 10, 2, 2), dtype = np.int)

        self.__player = None
        self.__dealer = None

        self.__history_of_ordinary_importance_sampling = []
        self.__history_of_weight_importance_sampling = []

    def __get_game_result(self, players_sum, dealers_sum):
        reward = 0.0
        if players_sum > 21:
            if dealers_sum <= 21:
                reward = -1.0
        else:
            if dealers_sum > 21:
                reward = 1.0
            else:
                if players_sum > dealers_sum:
                    reward = 1.0
                elif players_sum < dealers_sum:
                    reward = -1.0    
        return reward

    def __reset(self):
        self.__history_Vs = []
        self.__Vs = np.zeros((10, 10, 2))
        self.__Vs_count = np.zeros((10,10,2), dtype = np.int)
        self.__Q = np.zeros((10, 10, 2, 2))
        self.__Q_count = np.zeros((10, 10, 2, 2), dtype = np.int)
        self.__history_of_ordinary_importance_sampling = []
        self.__history_of_weighted_importance_sampling = []

    def __is_state_first_appeared(self, state, trajectory):
        trj_len = len(trajectory)
        for j in range(0, trj_len):
            if trajectory[j].state == state:
                return False
        return True

    def __is_state_action_first_appeared(self, state, action, trajectory):
        trj_len = len(trajectory)
        for j in range(0, trj_len):
            if trajectory[j].state == state \
                or trajectory[j].action == action:
                return False
        return True


    def __update_Vs(self, state, G):
        sum_index = state.card_sum - 12
        dealers_showing_index = state.dealers_showing - 1
        usable_ace_index = int(state.used_ace_num > 0) # 0--no usable ace, 1--usable ace
        v = self.__Vs[sum_index][dealers_showing_index][usable_ace_index]
        self.__Vs_count[sum_index][dealers_showing_index][usable_ace_index] += 1
        n = self.__Vs_count[sum_index][dealers_showing_index][usable_ace_index]
        self.__Vs[sum_index][dealers_showing_index][usable_ace_index] = v + (G - v) / n

    def __update_Q(self, state, action, G):
        sum_index = state.card_sum - 12
        dealers_showing_index = state.dealers_showing - 1
        usable_ace_index = int(state.used_ace_num > 0) # 0--no usable ace, 1--usable ace
        q = self.__Q[sum_index][dealers_showing_index][usable_ace_index][action]
        self.__Q_count[sum_index][dealers_showing_index][usable_ace_index][action] += 1
        n = self.__Q_count[sum_index][dealers_showing_index][usable_ace_index][action]
        self.__Q[sum_index][dealers_showing_index][usable_ace_index][action] = q + (G - q) / n
        return self.__Q[sum_index][dealers_showing_index][usable_ace_index]

    def MC_on_policy(self, episodes = 500000, first_visit_enabled = False, record_indices = []):
        self.__reset()
        record_index = 0
        pb = ProgressBar().start()
        self.__player = Player()
        self.__dealer = Dealer()
        for episode_index in range(0, episodes):
            dealers_showing = self.__dealer.start_game()
            self.__dealer.play_game()
            self.__player.start_game(dealers_showing)
            self.__player.play_game()

            reward = self.__get_game_result(self.__player.sum, self.__dealer.sum)
            self.__player.set_game_result(reward)

            G = 0.0
            trj = self.__player.trajectory
            trj_len = len(trj)
            for i in range(trj_len - 1, -1, -1):
                G = G + trj[i].reward
                state = trj[i].state
                if first_visit_enabled:
                    if not self.__is_state_first_appeared(state, trj[:i]):
                        continue
                self.__update_Vs(state, G)
            
            if record_indices[record_index] == episode_index + 1:
                self.__history_Vs.append(self.__Vs.copy())
                record_index += 1

            pb.update(int(episode_index /episodes * 100))
        pb.update(100)

    def MC_ES_on_policy(self, episodes = 500000, first_visit_enabled = False):
        self.__reset()
        self.__player = Player_MC_ES()
        self.__dealer = Dealer()
        pb = ProgressBar().start()
        for episode_index in range(0, episodes):
            dealers_showing_card = self.__dealer.start_game()
            self.__dealer.play_game()
            self.__player.start_game(dealers_showing_card)
            self.__player.play_game()

            reward = self.__get_game_result(self.__player.sum, self.__dealer.sum)
            self.__player.set_game_result(reward)

            G = 0.0
            trj = self.__player.trajectory
            trj_len = len(trj)
            for i in range(trj_len - 1, -1, -1):
                G = G + trj[i].reward
                state = trj[i].state
                action = trj[i].action
                if first_visit_enabled:
                    if not self.__is_state_action_first_appeared(state, action, trj[:i]):
                        continue                
                Q_under_actions = self.__update_Q(state, action, G)
                pai = np.random.choice([act for act, q_under_action in enumerate(Q_under_actions) \
                    if q_under_action == np.max(Q_under_actions)])
                self.__player.set_policy(state, pai)
            pb.update(int(episode_index / episodes * 100))
        pb.update(100)
        
        print('used ace')
        print(self.__Q[:, :, 1, :])
        print(self.__player.policy[:, :, 1])
        print('no used ace')
        print(self.__Q[:, :, 0, :])
        print(self.__player.policy[:, :, 0])

    def MC_off_policy(self,  episodes = 10000):
        self.__reset()
        self.__dealer = Dealer()
        
        nominator_sum = 0
        denominator_sum = 0

        for episode_index in range(0, episodes):
            self.__dealer.start_game()
            self.__dealer.play_game()
            player_target = Player_target()
            player_behavior = Player_behavior()
            player_behavior.start_game()
            player_behavior.play_game()
            reward = self.__get_game_result(player_behavior.sum, self.__dealer.sum)
            player_behavior.set_game_result(reward)
            trajectory = player_behavior.trajectory

            pai = 1
            b = 1
            for trj_element in trajectory:
                if trj_element.action == player_target.get_action(trj_element.state):
                    b *= 0.5
                else:
                    pai = 0
                    break #once pai == 0, rho = pai / b == 0. It's not necessary to continue for loop.
            rho = pai / b
            nominator_sum += rho * reward
            denominator_sum += rho
            # Because s is the intial state, J(s) is equal to the iteration times for both first visit
            # and every visit method.
            self.__history_of_ordinary_importance_sampling.append(nominator_sum / (episode_index + 1))
            if denominator_sum == 0.0:
                self.__history_of_weighted_importance_sampling.append(0.0)
            else:
                self.__history_of_weighted_importance_sampling.append(nominator_sum / denominator_sum)
        
    def calculate_variance(self, episodes):
        X_average = -0.27726
        #X_average_square = (-0.27726)**2
        list_VarX_ordinary = np.zeros(episodes)
        list_VarX_weighted = np.zeros(episodes)
        pb = ProgressBar().start()
        for i in range(0, 100):
            self.MC_off_policy(episodes)
            for j in range(0, episodes):
                var_ordinary = (self.__history_of_ordinary_importance_sampling[j] - X_average)**2
                var_weighted = (self.__history_of_weighted_importance_sampling[j] - X_average)**2
                list_VarX_ordinary[j] = list_VarX_ordinary[j] + (var_ordinary - list_VarX_ordinary[j]) / (i + 1)
                list_VarX_weighted[j] = list_VarX_weighted[j] + (var_weighted - list_VarX_weighted[j]) / (i + 1)
            pb.update(i)
        pb.update(100)
        return list_VarX_ordinary, list_VarX_weighted


    @property
    def history_Vs(self):
        return self.__history_Vs

    @property
    def optimal_Vs(self):
        return np.max(self.__Q, axis = -1)
    
    @property
    def optimal_policy(self):
        return self.__player.policy

    @property 
    def history_of_ordinary_importance_sampling(self):
        return self.__history_of_ordinary_importance_sampling

    @property 
    def history_of_weighted_importance_sampling(self):
        return self.__history_of_weighted_importance_sampling

class CPlot:
    def __init__(self, supTitle):
        self.__X = np.arange(1, 11, 1)
        self.__Y = np.arange(12, 22, 1)
        self.__X_3d, self.__Y_3d = np.meshgrid(self.__X, self.__Y)
        self.__fig = mplt_pyplt.figure(figsize= mplt_pyplt.figaspect(0.8))
        self.__fig.suptitle(supTitle)

    def plot_Vs(self, Z, subplt_index, ax_title):
        ax = self.__fig.add_subplot(2, 2, subplt_index, projection = '3d', proj_type = "persp")
        ax.plot_surface(self.__X_3d, self.__Y_3d, Z)
        ax.set_proj_type('persp')
        ax.set_zlim(-1, 1)
        ax.set_xlabel('Dealer\'s showing')
        ax.set_ylabel('Player\'s sum')
        ax.set_title(ax_title)

    def plot_pai(self, Z, subplt_index, ax_title):
        axes = self.__fig.add_subplot(2, 2, subplt_index)
        sbn_fig = sbn.heatmap(np.flipud(Z), ax=axes, xticklabels = list(range(1, 11)), \
            yticklabels = list(reversed(range(12, 22))))
        sbn_fig.set_xlabel('Dealer\'s showing')
        sbn_fig.set_ylabel('Player\'s sum')
        sbn_fig.set_title(ax_title)   

    def plot_mean_square_error(self, err_ordinary, err_weighted):
        mplt_pyplt.plot(err_ordinary, label = 'ordinary importance sampling')
        mplt_pyplt.plot(err_weighted, label = 'weighted importance sampling')
        mplt_pyplt.xscale('log')
        mplt_pyplt.xlabel('Episodes (log scale)')
        mplt_pyplt.ylabel('Mean Square Error')
        mplt_pyplt.legend()
        
    def show(self):
        mplt_pyplt.show()

def test_MC_on_policy(first_visit):
    bj = Blackjack()
    bj.MC_on_policy(500000, first_visit, [10000, 500000])
    cplt = None   
    if first_visit:
        cplt = CPlot('Monte Carlo (on-policy, first visit)')
    else:
        cplt = CPlot('Monte Carlo (on-policy, every visit)')
    
    cplt.plot_Vs(bj.history_Vs[0][:, :, 1], 1, "Usable Ace\nAfter 10,000 episodes")
    cplt.plot_Vs(bj.history_Vs[1][:, :, 1], 2, "Usable Ace\nAfter 500,000 episodes")
    cplt.plot_Vs(bj.history_Vs[0][:, :, 0], 3, "No Usable Ace\nAfter 10,000 episodes")
    cplt.plot_Vs(bj.history_Vs[1][:, :, 0], 4, "No Usable Ace\nAfter 500,000 episodes")
    cplt.show()

def test_MC_ES_on_policy(first_visit):
    bj = Blackjack()
    bj.MC_ES_on_policy(1000000, first_visit)
    cplt = None
    if first_visit:
        cplt = CPlot('Monte Carlo ES (on-policy, first visit)')
    else:
        cplt = CPlot('Monte Carlo ES (on-policy, every visit)') 

    cplt.plot_pai(bj.optimal_policy[:, :, 1], 1, r"$\pi_*$"+"\nUsable Ace")
    cplt.plot_Vs(bj.optimal_Vs[:, :, 1], 2, r"$v_*$"+"\nUsable Ace")
    cplt.plot_pai(bj.optimal_policy[:, :, 0], 3, r"$\pi_*$"+"\nNo Usable Ace")
    cplt.plot_Vs(bj.optimal_Vs[:, :, 0], 4, r"$v_*$"+"\nNo Usable Ace")
    cplt.show()

def test_MC_off_policy():
   bj = Blackjack()
   trajectory_var_ordinary, trajectory_var_weighted = bj.calculate_variance(10000)
   cplt = CPlot('Mean square error of ordinary and weighted importance sampling')
   cplt.plot_mean_square_error(trajectory_var_ordinary, trajectory_var_weighted)
   cplt.show()




if __name__ == "__main__":
    test_MC_on_policy(True)
    test_MC_on_policy(False)
    test_MC_ES_on_policy(True)
    test_MC_ES_on_policy(False)
    test_MC_off_policy()
