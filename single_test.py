#!/usr/bin/env python
# -*- coding: utf-8 -*-

from flask import Flask, jsonify, request, abort, Response
from time import time
from uuid import uuid4
import json

import tflearn
import tensorflow as tf
import Ddz_Core
import Ddz_AI_Play
import numpy as np
import doudizhu
import Ddz_Log_Parser
from minmax_engine import start_engine


IS_CARDNUM_SCAL = True  # 简单的调整 出多牌的策略

poker_mapping = {'1': '1', '2': '2', '3': '3', '4': '4', '5': '5', '6': '6', '7': '7', '8': '8', '9': '9'
    , '10': '10', '11': 'J', '12': 'Q', '13': 'K', '14': 'A', '15': '2', '16': "Y", '17': "Z"}

file_loc = '../new0vsnew12data/'
models = []
for i in range(0, 3):
    tf.reset_default_graph()
    model = tflearn.DNN(doudizhu.cnn())
    model.load(file_loc + 'models' + str(i) + '/fc_model_classify_' + str(i) + '.tflearn')
    models.append(model)

def len_cards(arr):
    sum = 0
    for i in range(15) :
        sum += arr[i]
    return sum
#根据三家历史出牌加上自己手牌推测出牌库未出的牌,进而预测敌方剩余手牌（取最大的N张牌）
def enemy_card(rest_arr,len_of_enemy):
    arr = []
    for i in range(0,13):
        rest_arr[i] = 4 - rest_arr[i]
    rest_arr[13] = 1 - rest_arr[13]
    rest_arr[14] = 1 - rest_arr[14]
    rest_cards = Ddz_Core.num_list_to_cards(rest_arr)
    a = sorted(rest_cards)
    for i in range(0,len_of_enemy):
        arr.append(a[len(rest_cards)-1-i])
    return  arr

def index():
    return ''' 斗地主接口   <br>

    role  身份  0 地主 1 地主下  2 地主上 <br>


    手牌数据  一个长度15的数组，数组里的值是 张数   <br>
     [ 0,3,0,1,3,0,0,0 ,0,0,0,0,0, 0, 0] 表示 手牌 4 4 4 6 7 7 7  <br>
  牌面  3 4 5 6 7 8 9 10 J Q K A 2  小 大  <br>


    data 数据 : <br>
    # 1. 自己 当前手牌 <br>
    # 2. 自己 上把出牌 <br>
    # 3. 上家上把出牌 <br>
    # 4. 下家上把出牌 <br>
    # 5. 自己历史出牌 <br>
    # 6. 上家历史出牌 <br>
    # 7. 下家历史出牌 <br>
    7个长度15的数组拼成一个数组

    data=3, 2, 0, 1, 0, 2, 1, 1, 3, 1, 2, 1, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0



    '''



def ddz():
    #data = request.values.get('data', None)
    data=3, 2, 0, 1, 0, 2, 1, 1, 3, 1, 2, 1, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0
    role = 0#request.values.get('role', None)
    type = 0#request.values.get('type', None)

    if not data:
        abort(400)

    data = list(data)
    data_arr = []
    for d in data:
        if d != ',' and d != ' ':
            data_arr.append(int(d))
    if len(data_arr) > 7 * 15:
        data_arr = data_arr[0:7 * 15]

    ddz = Ddz_Core.Doudizhu()

    ai_arrs = []
    for i in range(0, 7):
        ai_arrs.append(data_arr[i * 15:(i + 1) * 15])

    # 角色
    if role == None:
        role = 0
    else:
        role = int(role)
    # 张数数组 转为  牌面数组
    ddz.users[role] = Ddz_Core.num_list_to_cards(ai_arrs[0])

    #敌方剩余手牌数，取牌库剩余最大的几张牌当作敌方手牌
    rest_arr = []
    for i in range(15):
        rest_arr.append(ai_arrs[0][i]+ai_arrs[4][i]+ai_arrs[5][i]+ai_arrs[6][i]) #根据自己手牌和历史总出牌计算牌库剩余牌
    if role == 0 :
        len_up = 17 - len_cards(ai_arrs[5])
        len_down = 17 - len_cards(ai_arrs[6])
        if len_up >= len_down:
            len_enemy = len_down
        else :
            len_enemy = len_up
    if role == 1 :
        len_enemy = 20 - len_cards(ai_arrs[5])
    if role == 2 :
        len_enemy = 20 -len_cards(ai_arrs[6])



    mustHand = False
    # 上家或上上家有效的那一手
    lastHandPokers = Ddz_Core.num_list_to_cards(ai_arrs[2])
    if sum(lastHandPokers) == 0:
        lastHandPokers = Ddz_Core.num_list_to_cards(ai_arrs[3])
        if sum(lastHandPokers) == 0:  # 连续两把都是过，必须出牌
            mustHand = True

    # 牌型
    if type == None:  # 没给的话，根据上把出牌， 找最大的 hand
        lasthand = ddz.oneHandMaxType(lastHandPokers)
        type = lasthand['type']
    else:  # 给定，找出上把 hand
        type = Ddz_Log_Parser.cardTypeToCOMB_TYPE(type)
        lasthand = ddz.oneHand(lastHandPokers, type)

    allHands = ddz.allCanGoHandsByPokeAndLast(ddz.users[role], lasthand)
    if mustHand:
        #主动出牌策略加入残局破解，自己手牌小于13张且敌人手牌小于4张时触发
        my_cards = Ddz_Core.num_list_to_cards(ai_arrs[0])
        enemy_cards = enemy_card(rest_arr,len_enemy)
        if len(my_cards) <13 and len_enemy <4:
            lorder_move = start_engine(lorder_cards=[poker_mapping[str(x)] for x in my_cards], farmer_cards=[poker_mapping[str(x)] for x in enemy_cards],farmer_move=[])
            if lorder_move != None:
                return jsonify(lorder_move)
            else:
                allHands.remove(Ddz_Core.HAND_PASS)
        else:
            allHands.remove(Ddz_Core.HAND_PASS)
    scores = []
    for hand in allHands:
        ai_input = np.array(data_arr + ddz.handoutToAi(hand)).reshape([8, 15, 1])
        pred = models[role].predict([ai_input])

        scal = len(hand['poker'])
        if scal == 0:
            scal = 1
        if not IS_CARDNUM_SCAL:
            scal = 1
        scores.append(pred[0][0] * scal)
        # scores.append(pred[0][0])

    idx = scores.index(max(scores))
    hand = (allHands[scores.index(max(scores))])
    hand.get('poker')
    #return jsonify(hand.get('poker'))




if __name__ == '__main__':
    ddz()
