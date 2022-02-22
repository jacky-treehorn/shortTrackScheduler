# -*- coding: utf-8 -*-
"""
Spyder Editor

Dies ist eine temporäre Skriptdatei.
"""
from random import Random
import copy
import pandas as pd
from pointsAllocator import allocatePoints, randomPenaltyAdvancementMaker
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
    resultGenerator = Random()

    for heatId, heat in heatDict.items():
        heat_ = copy.copy(heat['heat'])
        resultGenerator.shuffle(heat_)
        heat_ = randomPenaltyAdvancementMaker(heat_, resultGenerator)
        print('\n')
        print('Heat {0} result: {1}'.format(heatId, heat_))
        allocatePoints(heat_, raceProgram_.skaterDict, verbose = True)
    for skater_ in raceProgram_.skaterDict.values():
        skater_.averageResults()
    dfList = []
    for skater_ in raceProgram_.skaterDict.values():
        dfList.append({'skaterNum':skater_.skaterNum,
                        'averagePoints':skater_.averageResult,
                        'skaterName':skater_.name,
                        'skaterTeam':skater_.team})
    ranking = pd.DataFrame(dfList)
    print('\n')
    print('Rankings')
    print(ranking.sort_values(by=['averagePoints'], ascending = False))