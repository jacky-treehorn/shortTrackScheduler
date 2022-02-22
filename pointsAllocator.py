# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 20:19:34 2022

@author: rasta
"""
import copy
from random import Random

def allocatePoints(heatResult, skatersDict, verbose = False):
    fibonacci = [5, 3, 2, 1]
    while len(fibonacci) < len(heatResult):
        fibonacci.append(0)
    for i, skaterNum in enumerate(heatResult):
        if hasattr(skaterNum, 'keys'):            
            assert len(skaterNum) == 1, 'Error in heat: {}. Only 1 key-value pair allowed'.format(heatResult)
            snvalues = [x for x in skaterNum.values()]
            if snvalues[0].lower() in ['p', 'dns']:
                for skaterNum_ in skaterNum.keys():
                    if verbose:
                        if snvalues[0].lower() == 'p':
                            print('Skater {0} receives 0 points. <-- PENALTY'.format(skaterNum_))
                        if snvalues[0].lower() == 'dns':
                            print('Skater {0} receives 0 points. <-- DNS'.format(skaterNum_))
                continue
            if 'a'== snvalues[0].lower():
                for skaterNum_ in skaterNum.keys():
                    skatersDict[skaterNum_].points += fibonacci[1]*(len(heatResult) - 1)
                    if verbose:
                        print('Skater {0} receives {1} points. <-- ADVANCED'.format(skaterNum_, fibonacci[1]*(len(heatResult) - 1)))
            if 'dnf'== snvalues[0].lower():
                for skaterNum_ in skaterNum.keys():
                    skatersDict[skaterNum_].points += fibonacci[len(heatResult) - 1]*(len(heatResult) - 1)
                    if verbose:
                        print('Skater {0} receives {1} points. <-- DNF'.format(skaterNum_, fibonacci[len(heatResult) - 1]*(len(heatResult) - 1)))
            continue
        if i < len(fibonacci):
            skatersDict[skaterNum].points += fibonacci[i]*(len(heatResult) - 1)
            if verbose:
                print('Skater {0} receives {1} points.'.format(skaterNum, fibonacci[i]*(len(heatResult) - 1)))

def randomPenaltyAdvancementMaker(heat, randomizer = Random(), prob_threshold = 0.8):
    heatIn = copy.copy(heat)
    if prob_threshold > 1.0 or prob_threshold <= 0.0:
        return heat
    if len(heat) < 2:
        return heat
    prob = randomizer.random()
    if prob <= prob_threshold:
        return heat
    if prob > prob_threshold:
        penIndex = randomizer.randint(0, len(heat) - 1)
        advIndex = randomizer.randint(0, len(heat) - 1)
        while advIndex == penIndex:
            advIndex = randomizer.randint(0, len(heat) - 1)
        penSkaterNum = heat[penIndex]
        advSkaterNum = heat[advIndex]
        n_loop = 0
        while penSkaterNum in heat:
            if n_loop >= 1:
                print('Invalid heat, duplicated results: {}'.format(heat))
            heat.remove(penSkaterNum)
            n_loop += 1
        n_loop = 0
        while advSkaterNum in heat:
            if n_loop >= 1:
                print('Invalid heat, duplicated results: {}'.format(heat))
            heat.remove(advSkaterNum)
            n_loop += 1
        heat.append({penSkaterNum:'p'})
        heat.append({advSkaterNum:'a'})
        if len(heat) == len(heatIn):
            return heat     
        else:
            return heatIn