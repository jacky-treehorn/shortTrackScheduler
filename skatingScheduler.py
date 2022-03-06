# -*- coding: utf-8 -*-
"""
Spyder Editor

Dies ist eine temporäre Skriptdatei.
"""
from random import Random
import copy
import pandas as pd
from pointsAllocator import pointsAllocation, randomPenaltyAdvancementMaker
from schedule import raceProgram
            
if __name__ == "__main__":
    raceProgram_ = raceProgram(totalSkaters=20,
                               numRacesPerSkater=3,
                               heatSize=5,
                               considerSeeding = False,
                               fairStartLanes = True,
                               minHeatSize=3
                               )
    heatDict = raceProgram_.buildHeats(adjustAfterNAttempts = 1000)
    pa = pointsAllocation(raceProgram_.skaterDict,
                          verbose = True,
                          ratingMaximum = 100.0)
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
            else:
                if result.lower() == 'a':
                    heatTimes[key] = float(2) + 40.0   
                else:
                    heatTimes[key] = float(result) + 40.0
        print('\n')
        print('Heat {0} result: {1}'.format(heatId, heat_))
        pa.allocatePoints(heat_, heatTimes, heatId)
    for skater_ in raceProgram_.skaterDict.values():
        skater_.averageResults()
        skater_.calculateBestTime()
    dfList = []
    for skater_ in raceProgram_.skaterDict.values():
        dfList.append({'skaterNum':skater_.skaterNum,
                       'rating':skater_.averageResult,
                       'bestTime':skater_.bestTime,
                       'skaterName':skater_.name,
                       'skaterTeam':skater_.team})
    ranking = pd.DataFrame(dfList)
    print('\n')
    print('Rankings')
    print(ranking.sort_values(by=['rating', 'bestTime'], ascending = [False, True]))