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

VALIDARGS = {"totalSkaters": "int",
             "numRacesPerSkater": "int",
             "heatSize": "int",
             "considerSeeding": "bool",
             "fairStartLanes": "bool",
             "minHeatSize": "int",
             "participantNames": "filePath", # For calls from winsport, do this later
             "participantTeams": "filePath", # For calls from winsport, do this later
             "participantAgeGroup": "filePath", # For calls from winsport, do this later
             "participantSeeding": "filePath", # For calls from winsport, do this later
             "method": "str",
             "runSimulation": "bool",
             "winsportOutputFullPath": "str",
             "winsportEventName": "str"
}

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
    debug = False
    argDict = {}
    val = None
    key = None
    for runArg in sys.argv[::-1]:
        if key is not None and val is not None:
            argDict[key] = val
            key = None
            val = None
        if isinstance(runArg, str):
            if val is None:
                val = runArg
                continue
            if key is None and runArg.startswith("--") and runArg[2:] in VALIDARGS:
                key = runArg[2:]
            else:
                val = None
    convertedArgDict = {}
    for key, val in argDict.items():
        if VALIDARGS[key] == "int":
            convertedArgDict[key] = int(val)
        if VALIDARGS[key] == "bool":
            convertedArgDict[key] = True if val.lower() in ["1", "true"] else False
        if VALIDARGS[key] == "filePath":
            print("Not implemented yet")
        if VALIDARGS[key] == "str":
            convertedArgDict[key] = val

    if debug:
        convertedArgDict["totalSkaters"] = 22
        convertedArgDict["numRacesPerSkater"] = 4
        convertedArgDict["heatSize"] = 4
        convertedArgDict["considerSeeding"] = False
        convertedArgDict["fairStartLanes"] = True
        convertedArgDict["minHeatSize"] = 4
        convertedArgDict["method"] = "sgp"
        convertedArgDict["runSimulation"] = True
    method="sgp"
    if "method" in convertedArgDict and convertedArgDict["method"].lower() in ['sgp', 'random_search', 'minimize']:
        method = convertedArgDict["method"].lower()
        del convertedArgDict["method"]
    runSimulation = False
    if "runSimulation" in convertedArgDict and convertedArgDict["runSimulation"]:
        runSimulation = convertedArgDict["runSimulation"]
    winsportOutputFullPath = ""
    if "winsportOutputFullPath" in convertedArgDict:
        winsportOutputFullPath = convertedArgDict["winsportOutputFullPath"]
    raceProgram_ = raceProgram(printDetails=True,
                               cleanCalculationDetails=True,
                               **convertedArgDict
                               )
    try:
        heatDict = raceProgram_.buildHeats(adjustAfterNAttempts=2000,
                                        method=method,
                                        winsportOutputFullPath=winsportOutputFullPath)
    except:
        sys.exit(1)
    if len(heatDict) == 0:
        if winsportOutputFullPath == "":
            input('No suitable heat structure could be found, press any key to exit.')
        sys.exit(1)
    if not runSimulation:
        if winsportOutputFullPath == "":
            input('No simulation requested, press any key to exit')
        sys.exit(0)
    pa = pointsAllocation(raceProgram_.skaterDict,
                          verbose=True,
                          ratingMaximum=100.0)
    resultGenerator = Random()

    try:
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
    except:
        sys.exit(1)
    if winsportOutputFullPath == "":
        input("Press any key to exit")
    sys.exit(0)