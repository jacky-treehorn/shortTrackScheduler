# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 20:19:34 2022

@author: rasta
"""
import copy
from random import Random

class pointsAllocation(object):
    
    def __init__(self,
                 skatersDict: dict,
                 ratingMaximum: float = 100.0,
                 verbose: bool = False,
                 pointsScheme: str = 'linear',
                 nonFinishPlacings: list = ['a', 'p', 'dns', 'dnf']):
        self.ratingMaximum = ratingMaximum
        self.skatersDict = skatersDict
        self.verbose = verbose
        self.pointsGenerator = lambda heatSize, pos: 0.0
        if pointsScheme == 'linear':
            self.pointsGenerator = lambda heatSize, pos: (heatSize - pos)*self.ratingMaximum if pos <= heatSize else 0.0
        if pointsScheme == 'fibonacci':
            self.pointsGenerator = lambda pos: 34.0 if pos <= 1 else 21.0 if pos == 2 else max(0.0, self.pointsGenerator(pos - 2) - self.pointsGenerator(pos - 1))
        self.nonFinishPlacings = nonFinishPlacings
        self.standardPlacings = lambda heatSize: list(range(1, heatSize+1))
        self.possiblePlacingGenerator = lambda heatSize: self.standardPlacings(heatSize) + self.nonFinishPlacings
        self.defaultPointsGenerator = lambda heatSize: dict(zip(self.standardPlacings(heatSize), [self.pointsGenerator(heatSize, x) for x in self.standardPlacings(heatSize)]))
            
    def _checkPlausiblePlacings(self, 
                                heatResult: dict):
        heatSize = len(heatResult)
        placings = self.standardPlacings(heatSize)
        possiblePlacings = self.possiblePlacingGenerator(heatSize)
        for result in heatResult.values():
            assert result in possiblePlacings, 'Impossible placement: {0}, only placements {1} are allowed for this heat.'.format(result, possiblePlacings)
        skipPlacing = []
        for placement in placings:
            if len(skipPlacing) > 0:
                for skipPlace in skipPlacing:
                    assert not (skipPlace in heatResult.values()), 'Error in heat result, place {} may not exist in a result with a dead heat.'.format(skipPlace)
                if placement in skipPlacing:
                    continue 
            n_sharedPlacement = -1
            for result in heatResult.values():
                if type(result) != str:
                    if result == placement:
                        n_sharedPlacement += 1
            if n_sharedPlacement > 0:
                for start_p in range(placement+1, placement+1+n_sharedPlacement):
                    skipPlacing.append(start_p)
        for skipPlace in skipPlacing:
            while skipPlace in possiblePlacings:
                possiblePlacings.remove(skipPlace)
        for actualPlacing in heatResult.values():
            if type(actualPlacing) == str:
                assert actualPlacing.lower() in possiblePlacings, 'Impossible placement: {0}, only {1} is allowed for this heat.'.format(actualPlacing, possiblePlacings)
            else:
                assert actualPlacing in possiblePlacings, 'Impossible placement: {0}, only {1} is allowed for this heat.'.format(actualPlacing, possiblePlacings)
        sortedActualPlacing = []
        for result in heatResult.values():
            if type(result) == str:
                continue
            else:
                sortedActualPlacing.append(result)
        expectedPlacing = []
        i = 0
        j = i
        while len(expectedPlacing) < len(sortedActualPlacing):
            i += 1
            if i in skipPlacing:
                expectedPlacing.append(j)
                continue
            j = i
            expectedPlacing.append(j)
        for i, result in enumerate(sorted(sortedActualPlacing)):
            assert result == expectedPlacing[i], 'Unexpected placement result: {0}, expected: {1}'.format(result, expectedPlacing[i])

    def _getDefaultPoints(self,
                          heatResult: dict) -> dict:
        heatSize = len(heatResult)
        possiblePlacings = self.possiblePlacingGenerator(heatSize)
        defaultPointsIn = self.defaultPointsGenerator(heatSize)
        points = {}
    
        for placement in defaultPointsIn.keys():
            if placement in possiblePlacings:
                n_sharedPlacement = -1
                for result in heatResult.values():
                    if result == placement:
                        n_sharedPlacement += 1
                if n_sharedPlacement > 0:
                    sharedPoints = 0
                    for placement_ in defaultPointsIn.keys():
                        if ((placement_ <= placement + n_sharedPlacement) and 
                            (placement_ > placement - 1)):
                            sharedPoints += defaultPointsIn[placement_]
                    sharedPoints = sharedPoints / (n_sharedPlacement + 1)
                    points[placement] = sharedPoints
                else:
                    points[placement] = defaultPointsIn[placement]
        return points

    def allocatePoints(self,
                       heatResult: dict):
        self._checkPlausiblePlacings(heatResult)
        defaultPoints = self._getDefaultPoints(heatResult)
        n_encounters = max(0, len(heatResult) - 1)
        for i, (skaterNum, result) in enumerate(heatResult.items()):
            if type(result) == str:
                if self.verbose:
                    if result.lower() == 'p':
                        print('Skater {0} receives 0 points. <-- PENALTY'.format(skaterNum))
                    if result.lower() in ['dnf','dns']:
                        print('Skater {0} receives 0 points. <-- DNS/DNF'.format(skaterNum))
                if result.lower() == 'a':
                    self.skatersDict[skaterNum].points += defaultPoints[2]
                    if self.verbose:
                        print('Skater {0} is ADVANCED'.format(skaterNum))
            else:
                self.skatersDict[skaterNum].points += defaultPoints[result]
        # Update each skater's running average
        for skaterNum in heatResult.keys():
            self.skatersDict[skaterNum].updateRunningAverageResult(n_encounters)
            if self.verbose:
                print('Skater {0} rating: '.format(skaterNum), self.skatersDict[skaterNum].rating)

def randomPenaltyAdvancementMaker(heat: dict, 
                                  randomizer = Random(), 
                                  prob_threshold: float = 0.8) -> dict:
    heatIn = copy.copy(heat)
    defaultResults = list(range(1, len(heat) + 1))
    skaterNums = list(heat.keys())
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
        penSkaterNum = skaterNums[penIndex]
        advSkaterNum = skaterNums[advIndex]
        n_loop = 0
        while penSkaterNum in heat.keys():
            if n_loop >= 1:
                print('Invalid heat, duplicated results: {}'.format(heat))
            del heat[penSkaterNum]
            n_loop += 1
        n_loop = 0
        while advSkaterNum in heat:
            if n_loop >= 1:
                print('Invalid heat, duplicated results: {}'.format(heat))
            del heat[advSkaterNum]
            n_loop += 1
        heat[penSkaterNum]='p'
        heat[advSkaterNum]='a'
        heatOut = {}
        finishingOrder = 0
        for result_ in defaultResults:
            if result_ in heat.values():
                finishingOrder += 1
                for skaterNum, res in heat.items():
                    if res == result_:
                        break
                heatOut[skaterNum] = finishingOrder
        heatOut.update({penSkaterNum: 'p', advSkaterNum: 'a'})
        heat = heatOut
            
        if len(heat) == len(heatIn):
            return heat     
        else:
            return heatIn

if __name__ == '__main__':
    #testing Area:
    from participant import skater
    skatersDict = {}
    for i in range(1,7):
        skatersDict[i] = skater(skaterNum = i)
    pA = pointsAllocation(skatersDict, ratingMaximum = 100.0, verbose = True)
    heatResult = {1:1, 2:2, 3:2, 4:4, 5:'p', 6:'a'}
    defaultPoints = pA._getDefaultPoints(heatResult)
    pA.allocatePoints(heatResult)
    for skaterNum, skater_ in skatersDict.items():
        print(skater_.points, ' ', skater_.rating)
    print(defaultPoints)