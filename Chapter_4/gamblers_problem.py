#==================================================================
# Python3
# Copyright
# 2019 Ye Xiang (xiang_ye@outlook.com)
#==================================================================

import numpy as np
import matplotlib as mpltlib
import matplotlib.pyplot as mtplt

class gamblers_problem:
    def __init__(self, ph = 0.4, dicimal_num = 9):
        self.__Ph = ph
        self.__win_goal = 100
        self.__dicimal_num = 9
        self.__Vs = np.zeros(self.__win_goal + 1)
        # Actually the terminal state value has no impact on the result.
        # This line can be removed.
        self.__Vs[self.__win_goal] = 1  
        self.__policy = np.zeros(self.__win_goal, dtype = np.int32)
        self.__history_Vs = []

    def __get_reward(self, state):
        return 1 if state >= self.__win_goal else 0

    def __calculate_Vs(self, vs, state):    
        prob_win = self.__Ph
        prob_lose = 1 - self.__Ph
        max_vs = vs[state]
        for act in range(1, min(state, self.__win_goal - state) + 1):
            tempVs = prob_win * (self.__get_reward(state + act) + vs[state + act]) \
                + prob_lose * (self.__get_reward(state - act) + vs[state - act])
            if tempVs > max_vs:
                max_vs = tempVs
        return max_vs

    def value_iteration(self, bRecord_history):
        accuracy = pow(10, -self.__dicimal_num)
        delta = accuracy + 1
        self.__history_Vs.clear()
        while delta >= accuracy:
            delta = 0
            if bRecord_history:
                self.__history_Vs.append(self.__Vs.copy())
            vs = self.__Vs.copy()
            for i in range(1, self.__win_goal):
                v = self.__Vs[i]
                self.__Vs[i] = self.__calculate_Vs(vs, i)
                delta = max(delta, abs(v - self.__Vs[i]))
            print('delta = {}'.format(delta))

    def policy_improvement(self):
        prob_win = self.__Ph
        prob_lose = 1 - self.__Ph
        for i in range(1, self.__win_goal):
            max_vs = np.round(self.__Vs[i], self.__dicimal_num)
            max_actions = []
            for act in range(1, min(i, self.__win_goal - i) + 1):
                tempVs = np.round(prob_win * (self.__get_reward(i + act) + self.__Vs[i + act]) \
                    + prob_lose * (self.__get_reward(i - act) + self.__Vs[i - act]), self.__dicimal_num)           
                if tempVs >= max_vs: # Here ">=" is used to get the argmax action. If only use ">", the argmax action 
                    max_vs = tempVs  # may not be updated which will lead to unchanged status.
                    if tempVs > max_vs:
                        max_actions.clear()
                    max_actions.append(act)
            self.__policy[i] = max_actions[0] # With ties broken arbitarily, in order to get the result as the book, 
                                              # the first action leads to maximum Vs is selected. The optimal policy
                                              # is not unique. If you select any other argmax action, the policy will
                                              # be changed.

    @property
    def policy(self):
        return self.__policy

    @property
    def Vs(self):
        return self.__Vs

    @property
    def history_Vs(self):
        return self.__history_Vs

    @property
    def win_goal(self):
        return self.__win_goal

def run_gambler(prob_head = 0.4, dicimal_num = 9):
    gp = gamblers_problem(prob_head, dicimal_num)
    gp.value_iteration(True)
    gp.policy_improvement()
    fig, axes = mtplt.subplots(2, 1, figsize = (15, 8))
    axes = axes.flatten()
    mtplt.suptitle(r'$P_h$ = {0}, $\theta$ = {1}'.format(prob_head, pow(10, -dicimal_num)))
    x = list(range(0, gp.win_goal))

    history_Vs_num = len(gp.history_Vs)
    for i in range(0, history_Vs_num):
        axes[0].plot(x, gp.history_Vs[i][0:gp.win_goal], label=r'$value${}'.format(i))
    axes[0].set_xlabel('Capital')
    axes[0].set_ylabel('Value estimates')
    #axes[0].legend()

    axes[1].plot(x, gp.policy, label=r'$\pi_*$')
    axes[1].set_xlabel('Capital')
    axes[1].set_ylabel('Policy (stakes)')
    axes[1].legend()
    print('policy = {}'.format(gp.policy))
    print('Vs = {}'.format(gp.Vs))
    mtplt.show()
    mtplt.close()

def run_gambler_2(prob_head = 0.4, dicimal_num_list = []):
    list_len = len(dicimal_num_list)
    if list_len < 1:
        return
    fig, axes = mtplt.subplots(list_len, 1, figsize = (15, 4 * list_len))
    mtplt.subplots_adjust(left = 0.05, right = 0.95, bottom = 0.05, top = 0.95, wspace=0.2, hspace=0.3)
    axes = axes.flatten()

    for i in range(0, list_len):    
        gp = gamblers_problem(prob_head, dicimal_num_list[i])
        gp.value_iteration(False)
        gp.policy_improvement()
        x = list(range(0, gp.win_goal))
        axes[i].plot(x, gp.policy, label=r'$\pi_*$')
        axes[i].set_xlabel(r'Capital  ($p_h$ = {0}, $\theta$ = {1})'.format(prob_head, pow(10,-dicimal_num_list[i])))
        axes[i].set_ylabel('Policy (stakes)')
        axes[i].legend()

    mtplt.show()
    mtplt.close()

if __name__ == "__main__":
    
    run_gambler(0.4, 9)
    run_gambler(0.25, 9)
    run_gambler(0.55, 9)
    '''
    run_gambler_2(0.25, [1,3,5,7,9,11])
    '''