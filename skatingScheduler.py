# -*- coding: utf-8 -*-
"""
Spyder Editor

Dies ist eine temporäre Skriptdatei.
"""
# pylint: disable=invalid-name
from random import Random
import copy
from pointsAllocator import pointsAllocation, randomPenaltyAdvancementMaker
from schedule import raceProgram

if __name__ == "__main__":
    raceProgram_ = raceProgram(totalSkaters=13,
                               numRacesPerSkater=4,
                               heatSize=4,
                               considerSeeding=True,
                               fairStartLanes=True,
                               minHeatSize=3,
                               printDetails=True,
                               cleanCalculationDetails=True
                               )
    heatDict = raceProgram_.buildHeats(adjustAfterNAttempts=1000)
    pa = pointsAllocation(raceProgram_.skaterDict,
                          verbose=True,
                          ratingMaximum=100.0)
    resultGenerator = Random()

    for heatId, heat in heatDict.items():
        heat_ = copy.copy(heat['heat'])
        resultGenerator.shuffle(heat_)
        heat_ = dict(zip(heat_, list(range(1, 1+len(heat_)))))
        heat_ = randomPenaltyAdvancementMaker(heat_, resultGenerator)
        heatTimes = {}
        for key, result in heat_.items():
            if result in pa.noTimePlacings:
                continue
            if result in ['a', 'A']:
                heatTimes[key] = float(2) + 40.0
            else:
                heatTimes[key] = float(result) + 40.0
        print('\n')
        print('Heat {0} result: {1}'.format(heatId, heat_))
        pa.allocatePoints(heat_, heatTimes, heatId)
        print('Intermediate results:\n')
        raceProgram_.buildResultsTable(
            intermediate=True, intermediatePrint=True, heatId=heatId)
    resultsTable = raceProgram_.buildResultsTable()
