# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 20:19:34 2022

@author: rasta
"""
# pylint: disable=invalid-name
import copy
from random import Random


class pointsAllocation():
    """ Allocates points to a skater after receiving heat results """

    def __init__(self,
                 skatersDict: dict,
                 ratingMaximum: float = 100.0,
                 verbose: bool = False,
                 pointsScheme: str = 'linear',
                 nonFinishPlacings: list = ['a', 'p', 'dns', 'dnf', 'y'],
                 noTimePlacings: list = ['p', 'dns', 'dnf', 'y']):
        self.ratingMaximum = ratingMaximum
        self.skatersDict = skatersDict
        self.verbose = verbose
        self.pointsGenerator = lambda pos, heatSize=0: 0.0
        if pointsScheme == 'linear':
            self.pointsGenerator = lambda pos, heatSize=4: (
                heatSize - pos)*self.ratingMaximum if pos <= heatSize else 0.0
        if pointsScheme == 'fibonacci':
            self.pointsGenerator = lambda pos, heatSize=0: 34.0 if pos <= 1 else 21.0 if pos == 2 else max(
                0.0, self.pointsGenerator(pos - 2) - self.pointsGenerator(pos - 1))
        self.nonFinishPlacings = nonFinishPlacings
        self.noTimePlacings = noTimePlacings
        self.standardPlacings = lambda heatSize: list(range(1, heatSize+1))
        self.possiblePlacingGenerator = lambda heatSize: self.standardPlacings(
            heatSize) + self.nonFinishPlacings
        self.defaultPointsGenerator = lambda heatSize: dict(zip(self.standardPlacings(
            heatSize), [self.pointsGenerator(x, heatSize) for x in self.standardPlacings(heatSize)]))
        self.cumulativeResults = []

    def _checkPlausiblePlacings(self,
                                heatResult: dict):
        heatSize = len(heatResult)
        placings = self.standardPlacings(heatSize)
        possiblePlacings = self.possiblePlacingGenerator(heatSize)
        for result in heatResult.values():
            assert result in possiblePlacings, \
                'Impossible placement: {0}, only placements {1} are allowed for this heat.'.format(
                    result, possiblePlacings)
        skipPlacing = []
        for placement in placings:
            if len(skipPlacing) > 0:
                for skipPlace in skipPlacing:
                    assert not (skipPlace in heatResult.values()), \
                        'Error in heat result, place {} may not exist in a result with a dead heat.'.format(
                            skipPlace)
                if placement in skipPlacing:
                    continue
            n_sharedPlacement = -1
            for result in heatResult.values():
                if not isinstance(result, str):
                    if result == placement:
                        n_sharedPlacement += 1
            if n_sharedPlacement > 0:
                for start_p in range(placement+1, placement+1+n_sharedPlacement):
                    skipPlacing.append(start_p)
        for skipPlace in skipPlacing:
            while skipPlace in possiblePlacings:
                possiblePlacings.remove(skipPlace)
        for actualPlacing in heatResult.values():
            if isinstance(actualPlacing, str):
                assert actualPlacing.lower() in possiblePlacings, \
                    'Impossible placement: {0}, only {1} is allowed for this heat.'.format(
                    actualPlacing, possiblePlacings)
            else:
                assert actualPlacing in possiblePlacings, \
                    'Impossible placement: {0}, only {1} is allowed for this heat.'.format(
                        actualPlacing, possiblePlacings)
        sortedActualPlacing = []
        for result in heatResult.values():
            if isinstance(result, str):
                continue
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
            assert result == expectedPlacing[i], \
                'Unexpected placement result: {0}, expected: {1}'.format(
                result, expectedPlacing[i])

    def _getDefaultPoints(self,
                          heatResult: dict) -> dict:
        """ Generates the default set of points for a heat. """
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
                        if placement - 1 < placement_ <= placement + n_sharedPlacement:
                            sharedPoints += defaultPointsIn[placement_]
                    sharedPoints = sharedPoints / (n_sharedPlacement + 1)
                    points[placement] = sharedPoints
                else:
                    points[placement] = defaultPointsIn[placement]
        return points

    def allocatePoints(self,
                       heatResult: dict,
                       heatTimes: dict,
                       heatNum: int) -> list:
        '''Allocates points after each heat.'''
        for skaterNum, result in heatResult.items():
            self.skatersDict[skaterNum].addHeatTime(heatNum)
            if result in self.noTimePlacings:
                continue
            if skaterNum in heatTimes.keys():
                self.skatersDict[skaterNum].addHeatTime(
                    heatNum, heatTimes[skaterNum])
        yellowCards = []
        if len(heatResult) <= 1:
            return yellowCards

        self._checkPlausiblePlacings(heatResult)
        defaultPoints = self._getDefaultPoints(heatResult)
        n_encounters = max(0, len(heatResult) - 1)
        if self.verbose:
            print('\n')
            print('heat: ', heatNum)
            print('\n')
        for skaterNum, result in heatResult.items():
            if isinstance(result, str):
                if result.lower() == 'y':
                    yellowCards.append(skaterNum)
                    if self.verbose:
                        print(
                            'Skater {0} receives 0 points. <-- YC'.format(skaterNum))
                if self.verbose:
                    if result.lower() == 'p':
                        print(
                            'Skater {0} receives 0 points. <-- PENALTY'.format(skaterNum))
                    if result.lower() in ['dnf', 'dns']:
                        print(
                            'Skater {0} receives 0 points. <-- DNS/DNF'.format(skaterNum))
                if result.lower() == 'a':
                    self.skatersDict[skaterNum].points += defaultPoints[2]
                    if self.verbose:
                        print('Skater {0} is ADVANCED'.format(skaterNum))
            else:
                self.skatersDict[skaterNum].points += defaultPoints[result]
        # Update each skater's running average
        for skaterNum in heatResult.keys():
            if skaterNum in yellowCards:
                continue
            self.skatersDict[skaterNum].updateRunningAverageResult(
                n_encounters)
            if self.verbose:
                print('Skater {0} rating: {1}, points: {2}'.format(skaterNum,
                                                                   self.skatersDict[skaterNum].rating,
                                                                   self.skatersDict[skaterNum].points))
        self.cumulativeResults.append({"heatResult": heatResult,
                                       "heatTimes": heatTimes,
                                       "heatNum": heatNum})
        for yc in yellowCards:
            for i, res in enumerate(self.cumulativeResults):
                if yc in res["heatResult"].keys():
                    if res["heatResult"][yc] not in self.nonFinishPlacings:
                        hr = copy.copy(res["heatResult"])
                        for skNum, res0 in hr.items():
                            if res0 in self.nonFinishPlacings:
                                continue
                            if res0 > res["heatResult"][yc]:
                                self.cumulativeResults[i]["heatResult"][skNum] -= 1
                    # Turn yellow cards into penalties
                    self.cumulativeResults[i]["heatResult"][yc] = 'p'
                if yc in res["heatTimes"].keys():
                    del self.cumulativeResults[i]["heatTimes"][yc]
        return yellowCards


def randomPenaltyAdvancementMaker(heat: dict,
                                  randomizer=Random(),
                                  prob_threshold: float = 0.8) -> dict:
    """ Function to generate some random results for testing purposes. """
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
        ycProb = randomizer.random()
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
        heat[penSkaterNum] = 'p'
        if ycProb > prob_threshold:
            heat[penSkaterNum] = 'y'
        heat[advSkaterNum] = 'a'
        heatOut = {}
        finishingOrder = 0
        for result_ in defaultResults:
            if result_ in heat.values():
                finishingOrder += 1
                for skaterNum, res in heat.items():
                    if res == result_:
                        break
                heatOut[skaterNum] = finishingOrder
        heatOut.update(
            {penSkaterNum: heat[penSkaterNum], advSkaterNum: heat[advSkaterNum]})
        heat = heatOut

        if len(heat) == len(heatIn):
            return heat
        return heatIn


if __name__ == '__main__':
    # testing Area:
    from participant import skater
    skatersDict_ = {}
    for ii in range(1, 7):
        skatersDict_[ii] = skater(skaterNum=ii)
    pA = pointsAllocation(skatersDict_, ratingMaximum=100.0, verbose=True)
    heatResult_ = {1: 1, 2: 2, 3: 2, 4: 4, 5: 'p', 6: 'a'}
    heatTimes_ = {1: 40.0, 2: 40.1, 3: 40.2, 4: 40.3}
#    defaultPoints_ = pA._getDefaultPoints(heatResult_)
    pA.allocatePoints(heatResult_, heatTimes_, 1)
    for skater_ in skatersDict_.values():
        print(skater_.points, ' ', skater_.rating)
#    print(defaultPoints_)
