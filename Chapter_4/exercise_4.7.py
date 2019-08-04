#==================================================================
# Python3
# Copyright
# 2019 Ye Xiang (xiang_ye@outlook.com)
#==================================================================

import car_rental as crl
import matplotlib as mpltlib
import matplotlib.pyplot as mtplt
import seaborn as sbn
import numpy as np
import math

class car_rental_ex_4_7(crl.car_rental):
    def __init__(self):
        crl.car_rental.__init__(self)
        self.__max_car_num_per_parking_lot = 10
        self.__parking_cost = 4

    def _get_action_cost(self, action):
        if action == -1:
            return 0
        else:
            return abs(action) * self._transfer_cost_per_car

    def __get_parking_cost(self, car_num):
        if car_num > self.__max_car_num_per_parking_lot:
            return self.__parking_cost
        else:
            return 0
    

    def _calculate_Vs_by_policy(self, Vs,car_num1_in_evening, car_num2_in_evening, action):
        tmp_vs = 0.0
        prob_of_requested_car_num1 = 0
        prob_of_requested_car_num2 = 0
        prob_of_returned_car_num1 = 0
        prob_of_returned_car_num2 = 0
        car_num1_in_next_morning = max(0, min(car_num1_in_evening - action, self._max_car_num))
        car_num2_in_next_morning = max(0, min(car_num2_in_evening + action, self._max_car_num))
        rented_car_num1 = 0
        rented_car_num2 = 0
        car_num1_in_next_evening = 0
        car_num2_in_next_evening = 0
        action_cost = self._get_action_cost(action)
        parking_cost1 = self.__get_parking_cost(car_num1_in_next_morning)
        parking_cost2 = self.__get_parking_cost(car_num2_in_next_morning)
     
        for requested_car_num1 in range(0, car_num1_in_next_morning + 1):
            if requested_car_num1 >= car_num1_in_next_morning:
                #This case stands for all the request car numbers larger then the car number in next morning.
                prob_of_requested_car_num1 = self._req_over_boundary_prob1[car_num1_in_next_morning]
            else:
                prob_of_requested_car_num1 = self._requested_prob1[requested_car_num1]
            rented_car_num1 = min(requested_car_num1, car_num1_in_next_morning)

            for requested_car_num2 in range(0, car_num2_in_next_morning + 1):
                if requested_car_num2 >= car_num2_in_next_morning:
                    #This case stands for all the request car numbers larger then the car number in next morning.
                    prob_of_requested_car_num2 = self._req_over_boundary_prob2[car_num2_in_next_morning]
                else:
                    prob_of_requested_car_num2 = self._requested_prob2[requested_car_num2]
                rented_car_num2 = min(requested_car_num2,car_num2_in_next_morning)
                reward = self._get_rental_reward(requested_car_num1, car_num1_in_next_morning) + self._get_rental_reward(requested_car_num2, car_num2_in_next_morning) \
                    - action_cost - parking_cost1 - parking_cost2
                
                ret_num1_to_boundary = self._max_car_num - (car_num1_in_next_morning - rented_car_num1)
                for returned_car_num1 in range(0, ret_num1_to_boundary + 1):
                    if returned_car_num1 >= ret_num1_to_boundary:
                        #This case stands for all the returned numbers larger then ret_num1_to_boundary
                        prob_of_returned_car_num1 = self._ret_over_boundary_prob1[ret_num1_to_boundary]
                    else:
                        prob_of_returned_car_num1 = self._returned_prob1[returned_car_num1]
                    car_num1_in_next_evening = min(car_num1_in_next_morning - rented_car_num1 + returned_car_num1, self._max_car_num)

                    ret_num2_to_boundary = self._max_car_num - (car_num2_in_next_morning - rented_car_num2)
                    for returned_car_num2 in range(0, ret_num2_to_boundary + 1):                           
                        if returned_car_num2 >= ret_num2_to_boundary:
                            #This case stands for all the returned numbers larger then ret_num2_to_boundary
                            prob_of_returned_car_num2 = self._ret_over_boundary_prob2[ret_num2_to_boundary]
                        else:
                            prob_of_returned_car_num2 = self._returned_prob2[returned_car_num2]       
                        car_num2_in_next_evening = min(car_num2_in_next_morning - rented_car_num2 + returned_car_num2, self._max_car_num)
                            
                        prob = prob_of_requested_car_num1 * prob_of_requested_car_num2 \
                            * prob_of_returned_car_num1 * prob_of_returned_car_num2
                        tmp_vs += prob * (reward + self._gamma * Vs[car_num1_in_next_evening][car_num2_in_next_evening])
               
        return tmp_vs

if __name__ == '__main__':
    
    car_rental_instance = car_rental_ex_4_7()
    car_rental_instance.policy_iteration(True)
    history_policy = car_rental_instance.history_policy
    value = car_rental_instance.Vs
    policy_num = len(history_policy)
    row = math.ceil((policy_num + 1) / 3)
    fig, axes = mtplt.subplots(row, 3, figsize=(20, row * 5))
    mtplt.subplots_adjust(left = 0.05, right = 0.95, bottom = 0.05, top = 0.95, wspace=0.2, hspace=0.3)
    axes = axes.flatten()
    
    for i in range(0, policy_num):
        sbn_fig = sbn.heatmap(np.flipud(history_policy[i]), ax=axes[i])
        sbn_fig.set_ylabel("cars at first location")
        sbn_fig.set_xlabel("cars at second location")
        sbn_fig.set_title(r"$\pi${}".format(i))
        sbn_fig.set_yticks(list(reversed(range(0, car_rental_instance.max_car_num + 1))))
        sbn_fig.tick_params(labelsize = 8)

    sbn_fig = sbn.heatmap(np.flipud(value), ax=axes[-1], cmap="YlGnBu")
    sbn_fig.set_ylabel("cars at first location")
    sbn_fig.set_xlabel("cars at second location")
    sbn_fig.set_title(r"$v_*$")
    sbn_fig.set_yticks(list(reversed(range(0, car_rental_instance.max_car_num + 1))))
    sbn_fig.tick_params(labelsize = 8)

    mtplt.show()
    mtplt.close()