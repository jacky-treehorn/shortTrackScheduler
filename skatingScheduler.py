# -*- coding: utf-8 -*-
"""
Spyder Editor

Dies ist eine temporäre Skriptdatei.
"""
# pylint: disable=invalid-name
from random import Random
import copy
import sys
from pointsAllocator import pointsAllocation, randomPenaltyAdvancementMaker
from schedule import raceProgram


def yellowCardReset(raceProgram_: raceProgram,
                    pointsAllocation_: pointsAllocation,
                    yellowCards: list,
                    heatId: int) -> dict:
    ''' If a skater receives a yellow card, the whole schedule
    must be recalculated, this is a convenience function to handle
    this situation.'''

    raceProgram_.handleYellowCards(yellowCards, heatId)
    cumulativeResults = copy.copy(pointsAllocation_.cumulativeResults)
    pointsAllocation_.cumulativeResults = []
    for resetHeatDict in cumulativeResults:
        pointsAllocation_.allocatePoints(**resetHeatDict)
    return raceProgram_.heatDict


if __name__ == "__main__":
    raceProgram_ = raceProgram(totalSkaters=18,
                               numRacesPerSkater=3,
                               heatSize=5,
                               considerSeeding=False,
                               fairStartLanes=True,
                               minHeatSize=3,
                               printDetails=True,
                               cleanCalculationDetails=True
                               )

    heatDict = raceProgram_.buildHeats(adjustAfterNAttempts=2000,
                                       method='minimize')
    if len(heatDict) == 0:
        print('No suitable heat structure could be found, exiting.')
        sys.exit()
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
        yellowCards = pa.allocatePoints(heat_, heatTimes, heatId)
        if len(yellowCards) > 0:
            heatDict = yellowCardReset(
                raceProgram_, pa, yellowCards, heatId)
        print('Intermediate results:\n')
        raceProgram_.buildResultsTable(
            intermediate=True, intermediatePrint=True, heatId=heatId)
    resultsTable = raceProgram_.buildResultsTable()
