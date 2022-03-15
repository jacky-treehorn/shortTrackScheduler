# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 20:41:04 2022

@author: rasta
"""
# pylint: disable=invalid-name
import os
from datetime import datetime
from random import Random
import copy
import json
import logging
from participant import skater
import numpy as np
import pandas as pd
from random import shuffle
from scipy.optimize import minimize


def setup_logger(name: str,
                 log_file: str,
                 level=logging.INFO,
                 fmt: str = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                 datefmt: str = '%Y-%m-%d--%H:%M:%S',
                 verbose: bool = False):
    """To setup as many loggers as you want"""

    # If the logger exists, kill it.
    if name in logging.Logger.manager.loggerDict.keys():
        del logging.Logger.manager.loggerDict[name]

    logger = logging.getLogger(name)
    handler = logging.FileHandler(log_file)
    if verbose:
        console = logging.StreamHandler()
        console.setLevel(level)
        console.setFormatter(fmt=fmt)
        logger.addhandler(console)
    handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))

    logger.setLevel(level)
    logger.addHandler(handler)

    return logger


def initializeMatrix(nSkaters: int,
                     reqAppearances: int,
                     optimalHeatSize: int,
                     maxDims: tuple) -> np.array:
    """Builds your initial Heat Matrix"""
    initMat = -np.ones(maxDims, int)
    out = []
    skaterList = list(range(nSkaters))*reqAppearances
    np.random.shuffle(skaterList)
    heat = []
    for i in skaterList:
        heat.append(i)
        if len(heat) == optimalHeatSize:
            out.append(heat)
            heat = []
    if len(heat) != 0:
        while len(heat) < optimalHeatSize:
            heat.append(-1)
        out.append(heat)
    out = np.asarray(out)
    initMat[:out.shape[0], :out.shape[1]] = out
    return initMat


def kroneckerDelta(x: int,
                   y: int) -> float:
    """Calculates the kronecker delta"""
    out = 0
    if x == y:
        out = 1
    return float(out)


class convergenceTests():

    def __init__(self,
                 minHeatSize: int,
                 verbose: bool = True,
                 printDetails: bool = True,
                 averageSeeding: float = 0.0,
                 numRacesPerSkater: int = 0,
                 sampleStdDev: float = 0.0
                 ):
        self.minHeatSize = minHeatSize
        self.verbose = verbose
        self.printDetails = printDetails
        self.averageSeeding = averageSeeding
        self.numRacesPerSkater = numRacesPerSkater
        self.sampleStdDev = sampleStdDev

    def heatLengthTest(self,
                       heatDict: dict,
                       n_attempts: int = 0,
                       logger=None) -> int:

        if not all((len(heat_['heat']) >= self.minHeatSize for heat_ in heatDict.values())):
            if self.verbose:
                print('heatSizeError: Attempt {0} produced an unfavourable Heat structure, modifying...'.format(
                    n_attempts))
            if self.printDetails:
                if logger is not None:
                    logger.info(
                        'heatSizeError: Attempt %s produced an unfavourable Heat structure, modifying...', n_attempts)
            return 0
        if any((len(heat_['heat']) != len(set(heat_['heat'])) for heat_ in heatDict.values())):
            if self.verbose:
                print('heatUniquenessError: Attempt {0} produced an unfavourable Heat structure, modifying...'.format(
                    n_attempts))
            if self.printDetails:
                if logger is not None:
                    logger.info(
                        'heatUniquenessError: Attempt %s produced an unfavourable Heat structure, modifying...', n_attempts)
            return 0
        return 1

    def appearanceTest(self,
                       n_appearancesErrors: int,
                       skaterDict: dict,
                       n_attempts: int = 0,
                       logger=None) -> tuple:
        if not all((skater_.totalAppearances == self.numRacesPerSkater for skater_ in skaterDict.values())):
            if self.verbose:
                print('totalAppearancesError: Attempt {0} produced an unfavourable Heat structure, modifying...'.format(
                    n_attempts))
            if self.printDetails:
                if logger is not None:
                    logger.info(
                        'totalAppearancesError: Attempt %s produced an unfavourable Heat structure, modifying...', n_attempts)
            n_appearancesErrors += 1
            return 0, n_appearancesErrors
        return 1, n_appearancesErrors

    def encounterTest(self,
                      n_encounterErrors: int,
                      shift: int,
                      skaterDict: dict,
                      n_attempts: int = 0,
                      logger=None) -> tuple:
        allEncounters = [
            x.totalEncounters for x in skaterDict.values()]
        encountersError = False
        for i, enctr in enumerate(allEncounters):
            for j in range(i+1, len(allEncounters)):
                if np.abs(enctr - allEncounters[j]) > shift:
                    encountersError = True
                    break
        if encountersError:
            n_encounterErrors += 1
            if self.verbose:
                print('encountersError: Attempt {0} produced an unfavourable Heat structure, modifying...'.format(
                    n_attempts))
            if self.printDetails:
                if logger is not None:
                    logger.info(
                        'encountersError: Attempt %s produced an unfavourable Heat structure, modifying...', n_attempts)
            return 0, n_encounterErrors
        return 1, n_encounterErrors

    def seedingTest(self,
                    n_seedingErrors: int,
                    heatDict: dict,
                    n_attempts: int = 0,
                    logger=None) -> tuple:
        seedingErrors = False
        for heat in heatDict.values():
            if np.abs(heat['averageSeeding'] - self.averageSeeding) > self.sampleStdDev:
                seedingErrors = True
        if seedingErrors:
            n_seedingErrors += 1
            if self.verbose:
                print('seedingErrors: Attempt {0} produced an unfavourable Heat structure, modifying...'.format(
                    n_attempts))
            if self.printDetails:
                logger.info(
                    'seedingErrors: Attempt %s produced an unfavourable Heat structure, modifying...', n_attempts)
            return 0, n_seedingErrors
        return 1, n_seedingErrors


class raceProgram():
    """ A class to create a schedule for a single distance in
    a short track speed skating competition."""

    def __init__(self,
                 totalSkaters: int = 0,
                 numRacesPerSkater: int = 0,
                 heatSize: int = 2,
                 minHeatSize: int = 2,
                 considerSeeding: bool = False,
                 fairStartLanes: bool = False,
                 participantNames: list = [],
                 participantTeams: dict = {},
                 participantAgeGroup: dict = {},
                 participantSeeding: dict = {},
                 printDetails: bool = False,
                 cleanCalculationDetails: bool = False
                 ):
        logging.shutdown()
        self.printDetailsPath = os.path.join(os.getcwd(), 'calculationDetails')
        if cleanCalculationDetails:
            self._cleanCalculationDetails()
        self._dateTimeFormat = '%Y%m%d-%H%M%S'
        self.printDetails = printDetails
        self.totalSkaters = totalSkaters
        self.participantNames = participantNames
        if len(self.participantNames) > 0:
            assert len(self.participantNames) == self.totalSkaters, \
                'Length of participant names does not match the number of skaters!'
            assert len(self.participantNames) == len(set(self.participantNames)), \
                'Possible duplicate in participant names.'
        self.participantTeams = participantTeams
        if len(self.participantTeams) > 0:
            for key in self.participantTeams.keys():
                assert key in self.participantNames.keys(), \
                    '{} in participantTeams not found in participantNames'.format(
                        key)
        self.participantAgeGroup = participantAgeGroup
        if len(self.participantAgeGroup) > 0:
            for key in self.participantAgeGroup.keys():
                assert key in self.participantNames.keys(), \
                    '{} in participantAgeGroup not found in participantNames'.format(
                        key)
        self.participantSeeding = participantSeeding
        if len(self.participantSeeding) > 0:
            assert len(self.participantSeeding) == self.totalSkaters, \
                'Length of participant seeding does not match the number of skaters!'
            for key in self.participantSeeding.keys():
                assert key in self.participantNames.keys(),  \
                    '{} in participantSeeding not found in participantNames'.format(
                        key)
            assert set(self.participantSeeding.values()) == set(list(range(1, totalSkaters + 1))), \
                'Seeding should only contain sequential numbers from {0} to {1}'.format(
                    1, totalSkaters)
        self.heats = []
        self.numRacesPerSkater = numRacesPerSkater
        assert self.numRacesPerSkater >= 0, 'numRaces must be greater than or equal to 0.'
        self.heatSize = heatSize
        assert self.heatSize > 1, 'Heat size must be at least 2.'
        self.considerSeeding = considerSeeding
        self.fairStartLanes = fairStartLanes
        self.minHeatSize = minHeatSize
        assert self.minHeatSize > 1, 'Minimum heat size must be at least 2.'
        assert self.heatSize >= self.minHeatSize, \
            'Heat size must be at least {}.'.format(self.minHeatSize)
        self.skaterDict = {}
        self.randomizer = Random()
        self._laneValues = {0: 6.424, 1: 3.527, 2: 2.401, 3: 1.374, 4: 1.0,
                            5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0}
        lanevals = []
        for lane, val in self._laneValues.items():
            if lane >= self.heatSize:
                continue
            lanevals.append(val)
        self._laneAverage = np.mean(lanevals)
        self.heatOrder = []
        self.heatDict = {}
        self.startLaneStddev = 0.0
        self.resultsTable = None
        if self.printDetails:
            if not os.path.exists(self.printDetailsPath):
                os.mkdir(self.printDetailsPath)
                os.chmod(self.printDetailsPath, 0o777)
        self.startLaneLogger = None
        self.buildHeatsLogger = None
        if self.printDetails:
            self.startLaneLogger = setup_logger('startLanes',
                                                os.path.join(self.printDetailsPath,
                                                             'startLanes_' + datetime.now().strftime(self._dateTimeFormat)
                                                             + '.txt'))
            self.buildHeatsLogger = setup_logger('buildHeats',
                                                 os.path.join(self.printDetailsPath,
                                                              'buildHeats_' + datetime.now().strftime(self._dateTimeFormat)
                                                              + '.txt'))
        self.shift = 0
        self.n_encounterErrors = 0
        self.skaterNums = list(range(self.totalSkaters))
        self.averageSeeding = float(self.totalSkaters + 1) / 2.0
        self.sampleStdDev = np.std(self.skaterNums)/np.sqrt(self.heatSize)

    def _cleanCalculationDetails(self):
        if os.path.exists(self.printDetailsPath):
            filesToDelete = []
            foldersToDelete = []
            for root, dirs, files in os.walk(self.printDetailsPath):
                for fil in files:
                    os.chmod(os.path.join(root, fil), 0o777)
                    filesToDelete.append(os.path.join(root, fil))
                for dr in dirs:
                    os.chmod(os.path.join(root, dr), 0o777)
                    foldersToDelete.append(os.path.join(root, dr))
            for fileToDelete in filesToDelete:
                os.remove(fileToDelete)
            for folderToDelete in foldersToDelete:
                os.rmdir(folderToDelete)

    def buildResultsTable(self,
                          intermediate: bool = False,
                          intermediatePrint: bool = False,
                          verbose: bool = True,
                          heatId: str = '0'):
        """ Calculates the results of all the competitors """
        for skater_ in self.skaterDict.values():
            skater_.averageResults()
            skater_.calculateBestTime()
        dfList = []
        for skater_ in self.skaterDict.values():
            rating = skater_.averageResult
            if intermediate:
                rating = skater_.rating
            dfList.append({'skaterNum': skater_.skaterNum,
                           'rating': rating,
                           'bestTime': skater_.bestTime,
                           'skaterName': skater_.name,
                           'skaterTeam': skater_.team,
                           'encounters': skater_.cumulativeEncounters})
        ranking = pd.DataFrame(dfList)
        self.resultsTable = ranking[ranking['encounters'] > 0].sort_values(by=['rating', 'bestTime'],
                                                                           ascending=[False, True])
        self.resultsTable.drop(columns=['encounters'], inplace=True)
        if verbose:
            print('Rankings')
            print(self.resultsTable)
        if intermediatePrint:
            self.resultsTable.to_csv(os.path.join(self.printDetailsPath,
                                     'results_intermediate_heat_{0}_'.format(heatId)+datetime.now().strftime(self._dateTimeFormat)+'.csv'), sep=';')
        if not intermediate and self.printDetails:
            self.resultsTable.to_csv(os.path.join(self.printDetailsPath,
                                     'results_'+datetime.now().strftime(self._dateTimeFormat)+'.csv'), sep=';')
        return self.resultsTable

    def _appearancePotential(self,
                             heatMat: np.array) -> float:
        """Calculates the appearance potential"""
        pot = 0
        val, counts = np.unique(
            np.rint(heatMat.flatten()).astype(int), return_counts=True)
        binCountInit = dict(
            zip(range(self.totalSkaters), [0]*self.totalSkaters))
        binCount = dict(zip(val, counts))
        for skater_, count in binCountInit.items():
            if skater_ in binCount.keys():
                continue
            binCount[skater_] = count
        for skater_n in range(self.totalSkaters):
            count = 0
            count = binCount[skater_n]
            pot += (self.numRacesPerSkater - count)**2
        return float(pot)

    def _encounterPotential(self,
                            heatMatIn: np.array,
                            maxTheoreticalMatrixDims: tuple) -> float:
        """Calculates the encounter potential"""
        expectedEncounters = ((self.heatSize-1) *
                              self.numRacesPerSkater) - self.n_encounterErrors
        expectedEncounters = max(expectedEncounters, self.totalSkaters//2)
        encounterFactor = expectedEncounters/self.totalSkaters
        heatMat = heatMatIn
        if heatMatIn.ndim == 1:
            heatMat = heatMatIn.reshape(maxTheoreticalMatrixDims)
        binCountRowWise = []
        for i in range(heatMat.shape[0]):
            val, counts = np.unique(
                np.rint(heatMat[i, :]).astype(int), return_counts=True)
            binCount = dict(zip(val, counts))
            binCountRowWise.append(binCount)
        pot = 0
        for i in range(self.totalSkaters):
            encounters = []
            for j in range(self.totalSkaters):
                if i == j:
                    continue
                for row in binCountRowWise:
                    if i in row.keys() and j in row.keys():
                        if encounters.count(j) > encounterFactor:
                            continue
                        encounters.append(j)
            pot += (expectedEncounters - len(encounters))**2
        return float(pot)

    def _heatSizePotential(self,
                           heatMatIn: np.array,
                           maxTheoreticalMatrixDims: tuple) -> float:
        """Calculates the heat size potential"""
        heatMat = heatMatIn
        if heatMatIn.ndim == 1:
            heatMat = heatMatIn.reshape(maxTheoreticalMatrixDims)
        skaters = list(range(self.totalSkaters))
        pot = 0
        for i in range(heatMat.shape[0]):
            rowSum = 0
            if all((x < 0 for x in np.rint(heatMat[i, :]))):
                continue
            for skater_ in np.rint(heatMat[i, :]).astype(int):
                if skater_ in skaters:
                    rowSum += 1
            pot += (rowSum - self.heatSize)**2
        return float(pot)

    def _heatUniquenessPotential(self,
                                 heatMatIn: np.array,
                                 maxTheoreticalMatrixDims: tuple) -> float:
        """Calculates the uniqueness potential"""
        heatMat = heatMatIn
        if heatMatIn.ndim == 1:
            heatMat = heatMatIn.reshape(maxTheoreticalMatrixDims)
        pot = 0
        for i in range(heatMat.shape[0]):
            rowSum = 0
            for j in range(heatMat.shape[1]):
                if np.rint(heatMat[i, j]) < 0:
                    continue
                for k in range(j+1, heatMat.shape[1]):
                    if np.rint(heatMat[i, k]) < 0:
                        continue
                    rowSum += kroneckerDelta(
                        np.rint(heatMat[i, j]), np.rint(heatMat[i, k]))
            pot += rowSum
        return float(pot)

    def heatPotentialCalc(self,
                          heatMat: np.array,
                          maxTheoreticalMatrixDims: tuple) -> float:
        pot = self._appearancePotential(heatMat)
        pot += self._encounterPotential(heatMat,
                                        maxTheoreticalMatrixDims)
        pot += self._heatSizePotential(heatMat,
                                       maxTheoreticalMatrixDims)
        pot += self._heatUniquenessPotential(heatMat,
                                             maxTheoreticalMatrixDims)
        return pot

    def _randomSearch(self,
                      max_attempts: int = 10000,
                      adjustAfterNAttempts: int = 500,
                      encounterFlexibility: int = 0,
                      verbose: bool = False
                      ) -> dict:

        conTests = convergenceTests(minHeatSize=self.minHeatSize,
                                    verbose=verbose,
                                    printDetails=self.printDetails,
                                    averageSeeding=self.averageSeeding,
                                    numRacesPerSkater=self.numRacesPerSkater,
                                    sampleStdDev=self.sampleStdDev)
        heatDict = {}
        n_attempts = 1
        n_encounterErrors = 0
        n_personalEncounterErrors = 0
        n_seedingErrors = 0
        n_appearancesErrors = 0
        n_appearanceErrorsResets = 0
        shift = 0
        while True:
            if n_appearancesErrors > adjustAfterNAttempts:
                n_appearancesErrors = 0
                n_appearanceErrorsResets += 1
                if n_appearanceErrorsResets % 2:
                    n_encounterErrors = 0
                    shift += 1
                    if verbose:
                        print('Increasing shift: {}'.format(shift))
                    if self.printDetails:
                        self.buildHeatsLogger.info(
                            'Increasing shift: %s', shift)
                else:
                    n_personalEncounterErrors = 0
                    encounterFlexibility += 1
                    if verbose:
                        print('Increasing encounterFlexibility: {}'.format(
                            encounterFlexibility))
                    if self.printDetails:
                        self.buildHeatsLogger.info(
                            'Increasing encounterFlexibility: %s', encounterFlexibility)
            if n_encounterErrors > adjustAfterNAttempts:
                n_encounterErrors = 0
                shift += 1
                if verbose:
                    print('Increasing shift: {}'.format(shift))
                if self.printDetails:
                    self.buildHeatsLogger.info('Increasing shift: %s', shift)
            if n_personalEncounterErrors > adjustAfterNAttempts:
                n_personalEncounterErrors = 0
                encounterFlexibility += 1
                if verbose:
                    print('Increasing encounterFlexibility: {}'.format(
                        encounterFlexibility))
                if self.printDetails:
                    self.buildHeatsLogger.info(
                        'Increasing encounterFlexibility: %s', encounterFlexibility)
            if n_seedingErrors > adjustAfterNAttempts:
                n_seedingErrors = 0
                self.sampleStdDev *= 1.1
                if verbose:
                    print('Increasing sampleStdDev: {}'.format(self.sampleStdDev))
                if self.printDetails:
                    self.buildHeatsLogger.info(
                        'Increasing sampleStdDev: %s', self.sampleStdDev)
            if n_attempts >= max_attempts:
                print('No success after {} attempts. Quitting.'.format(n_attempts))
                if self.printDetails:
                    self.buildHeatsLogger.info(
                        'No success after %s attempts. Quitting.', n_attempts)
                return {}
            self.randomizer.shuffle(self.skaterNums)
            for i_skater in self.skaterNums:
                if self.skaterDict[i_skater].totalAppearances >= self.numRacesPerSkater:
                    continue
                heatNum = 0
                while True:
                    heatNum += 1
                    if heatNum in self.skaterDict[i_skater].heatAppearances:
                        continue
                    heat = []
                    if heatNum in heatDict.keys():
                        heat = heatDict[heatNum]['heat']
                    if self.skaterDict[i_skater].skaterNum in heat:
                        continue
                    if len(heat) >= self.heatSize:
                        continue
                    previouslyEncountered = False
                    for otherSkaterNum in heat:
                        otherSkater = self.skaterDict[otherSkaterNum]
                        if ((self.skaterDict[i_skater].skaterNum in otherSkater.encounters) or
                                (otherSkaterNum in self.skaterDict[i_skater].encounters)):
                            if ((self.skaterDict[i_skater].totalEncounters - self.skaterDict[i_skater].totalUniqueEncounters >= encounterFlexibility) or
                                    (otherSkater.totalEncounters - otherSkater.totalUniqueEncounters >= encounterFlexibility)):
                                previouslyEncountered = True
                                break
                    if previouslyEncountered:
                        n_personalEncounterErrors += 1
                        continue
                    for otherSkaterNum in heat:
                        self.skaterDict[i_skater].addEncounterFlexible(
                            otherSkaterNum)
                        self.skaterDict[otherSkaterNum].addEncounterFlexible(
                            i_skater)
                    self.skaterDict[i_skater].addHeatAppearance(heatNum)
                    heat.append(self.skaterDict[i_skater].skaterNum)
                    heatDict[heatNum] = {}
                    heatDict[heatNum]['heat'] = heat
                    heatDict[heatNum]['averageSeeding'] = 0
                    for skaterNum in heat:
                        heatDict[heatNum]['averageSeeding'] += self.skaterDict[skaterNum].seed
                    heatDict[heatNum]['averageSeeding'] /= len(heat)
                    if self.skaterDict[i_skater].totalAppearances >= self.numRacesPerSkater:
                        break
            heatsToDelete = []
            for heatNum, heat in heatDict.items():
                if len(heat['heat']) < self.minHeatSize:
                    heatsToDelete.append(heatNum)
                    for skaterNum_0 in heat['heat']:
                        self.skaterDict[skaterNum_0].removeHeatAppearance(
                            heatNum)
                        for skaterNum_1 in heat['heat']:
                            if skaterNum_1 != skaterNum_0:
                                self.skaterDict[skaterNum_0].removeEncounterFlexible(
                                    skaterNum_1)
                                self.skaterDict[skaterNum_1].removeEncounterFlexible(
                                    skaterNum_0)
            for heatNum in heatsToDelete:
                del heatDict[heatNum]

            tr = conTests.heatLengthTest(heatDict,
                                         n_attempts,
                                         logger=self.buildHeatsLogger)
            if tr == 0:
                self.reorganizeHeats(heatDict)
                continue

            longestHeatLength = 0
            for heatNum, heat in heatDict.items():
                if len(heat['heat']) > longestHeatLength:
                    longestHeatLength = len(heat['heat'])
            heatAsArray = np.zeros((len(heatDict), longestHeatLength))
            for heatInd, heat in enumerate(heatDict.values()):
                heatAsArray[heatInd, :len(heat['heat'])] = heat['heat']
            heatScore = self.heatPotentialCalc(heatAsArray, heatAsArray.shape)
            if verbose:
                print('Heat Score: {}'.format(heatScore))
            if self.printDetails:
                self.buildHeatsLogger.info('Heat Score: %s', heatScore)

            tr, n_appearancesErrors = conTests.appearanceTest(n_appearancesErrors,
                                                              self.skaterDict,
                                                              n_attempts,
                                                              logger=self.buildHeatsLogger)
            if tr == 0:
                self.reorganizeHeats(heatDict)
                continue
            n_attempts += 1
            tr, n_encounterErrors = conTests.encounterTest(n_encounterErrors,
                                                           shift,
                                                           self.skaterDict,
                                                           n_attempts,
                                                           logger=self.buildHeatsLogger)
            if tr == 0:
                shift += 1
                self.reorganizeHeats(heatDict)
                continue
            if self.considerSeeding:
                tr, n_seedingErrors = conTests.seedingTest(n_seedingErrors,
                                                           heatDict,
                                                           n_attempts,
                                                           logger=self.buildHeatsLogger)
                if tr == 0:
                    self.reorganizeHeats(heatDict)
                    continue
            if self.fairStartLanes:
                self.makeStartLanesFair(heatDict)
            print('Success after {} attempts.'.format(n_attempts))
            if self.printDetails:
                self.buildHeatsLogger.info(
                    'Success after %s attempts.', n_attempts)
                break
        self.shift = shift
        return heatDict

    def buildHeats(self,
                   max_attempts: int = 10000,
                   adjustAfterNAttempts: int = 500,
                   encounterFlexibility: int = 0,
                   verbose: bool = True,
                   method: str = 'random_search') -> dict:
        """ Calculates a heat structure """
        assert method in ['random_search', 'gradopt'], 'method must be either {}'.format(
            ['random_search', 'gradopt'])
        adjustAfterNAttempts = min(max_attempts, adjustAfterNAttempts)
        if self.numRacesPerSkater == 0:
            while self.totalSkaters / 2**self.numRacesPerSkater > self.heatSize:
                self.numRacesPerSkater += 1
        for i_skater in self.skaterNums:
            skaterName = 'Person_'+str(i_skater)
            if i_skater < len(self.participantNames):
                skaterName = self.participantNames[i_skater]
            seed = i_skater + 1
            if skaterName in self.participantSeeding.keys():
                seed = self.participantSeeding[skaterName]
            team = None
            if skaterName in self.participantTeams.keys():
                team = self.participantTeams[skaterName]
            ageCategory = None
            if skaterName in self.participantAgeGroup.keys():
                ageCategory = self.participantAgeGroup[skaterName]

            self.skaterDict[i_skater] = skater(i_skater,
                                               seed=seed,
                                               name=skaterName,
                                               team=team,
                                               ageCategory=ageCategory)

        heatDict = self._randomSearch(max_attempts,
                                      adjustAfterNAttempts,
                                      encounterFlexibility,
                                      verbose)
        if len(heatDict) == 0:
            return heatDict
        if self.shift > 1 or encounterFlexibility > 1:
            print(
                'WARNING! Some skaters may have noticably fewer encounters than others.')
            if self.printDetails:
                self.buildHeatsLogger.info(
                    'WARNING! Some skaters may have noticably fewer encounters than others. Shift = %s, EncounterFlexibility = %s', self.shift, encounterFlexibility)
        heatSpacing = self.spaceHeatsOut(heatDict)
        if len(heatSpacing) == 0:
            print(
                'WARNING!: No suitable heat spacing could be found.')
            if self.printDetails:
                self.buildHeatsLogger.info(
                    'WARNING!: No suitable heat spacing could be found.')
        if len(heatSpacing) > 0:
            if verbose:
                print('Optimal heat order: ', heatSpacing)
                print('Heats will be renumbered...')
            if self.printDetails:
                self.buildHeatsLogger.info(
                    'Optimal heat order: %s', heatSpacing)
                self.buildHeatsLogger.info('Heats will be renumbered...')
            self.heatOrder = [x for x in heatSpacing if isinstance(x, int)]
            heatDict_ = {}
            for skater_ in self.skaterDict.values():
                skater_.removeAllHeatAppearances()
            for i, heat in enumerate(self.heatOrder):
                heatDict_[i+1] = heatDict[heat]
                for skaterNum in heatDict[heat]['heat']:
                    self.skaterDict[skaterNum].addHeatAppearance(i+1)
            heatDict = heatDict_
        self.heatDict = heatDict
        for heatNum, heat in heatDict.items():
            printText = 'Heat {0}: {1}'.format(heatNum, heat['heat'])
            if self.considerSeeding:
                printText = 'Heat {0}: {1}, Seeding Check: {2}'.format(
                    heatNum, heat['heat'], np.abs(heat['averageSeeding'] - self.averageSeeding) < self.sampleStdDev)
            if verbose:
                print(printText)
            if self.printDetails:
                self.buildHeatsLogger.info(printText)
        for skater_ in self.skaterDict.values():
            if verbose:
                print('Skater {0} appears in {1} heats: '.format(skater_.skaterNum, skater_.totalAppearances), skater_.heatAppearances,
                      ', Total encounters: {0}, Total unique encounters: {1}'.format(skater_.totalEncounters, skater_.totalUniqueEncounters))
            if self.printDetails:
                self.buildHeatsLogger.info(
                    'Skater %s appears in %s heats: ', skater_.skaterNum, skater_.totalAppearances)
                self.buildHeatsLogger.info(
                    'Skater %s all appearances %s', skater_.skaterNum, skater_.heatAppearances)
                self.buildHeatsLogger.info('Skater %s, Total encounters: %s, Total unique encounters: %s',
                                           skater_.skaterNum, skater_.totalEncounters, skater_.totalUniqueEncounters)
            for heatNum_ in skater_.heatAppearances:
                assert skater_.skaterNum in self.heatDict[heatNum_]['heat'], \
                    'heatAllocationError: Skater {0} is not in the allocated heat: {1}'.format(
                        skater_.skaterNum, self.heatDict[heatNum_]['heat'])
        if self.printDetails:
            with open(os.path.join(self.printDetailsPath, 'heats_' +
                                   datetime.now().strftime(self._dateTimeFormat)+'.json'), 'w') as fil:
                json.dump(heatDict, fil, indent=4)

        return heatDict

    def reorganizeHeats(self, heatDict: dict):
        """ Randomly removes a competitor from each heat. """
        removedSkaterDict = {}
        for heatNum, heat in heatDict.items():
            if len(heat['heat']) >= min(3, self.heatSize):
                removedSkaterIndex = self.randomizer.randint(
                    0, len(heat['heat']) - 1)
                removedSkaterDict[heatNum] = heat['heat'][removedSkaterIndex]
        for heatNum, skaterNum in removedSkaterDict.items():
            self.skaterDict[skaterNum].removeHeatAppearance(heatNum)
            for skaterNum_0 in heatDict[heatNum]['heat']:
                if skaterNum_0 != skaterNum:
                    self.skaterDict[skaterNum].removeEncounterFlexible(
                        skaterNum_0)
                    self.skaterDict[skaterNum_0].removeEncounterFlexible(
                        skaterNum)
            while skaterNum in heatDict[heatNum]['heat']:
                heatDict[heatNum]['heat'].remove(skaterNum)

    def makeStartLanesFair(self,
                           heatDict: dict,
                           verbose: bool = True):
        """ Does what it says """
        skaterDict = {}
        for heatNum, heat in heatDict.items():
            for lane, skater_ in enumerate(heat['heat']):
                value = 0
                if lane in self._laneValues.keys():
                    value = self._laneValues[lane]
                if skater_ in skaterDict.keys():
                    skaterDict[skater_]['values'].append(value)
                    skaterDict[skater_]['heatNum'].append(heatNum)
                else:
                    skaterDict[skater_] = {'values': [value]}
                    skaterDict[skater_]['heatNum'] = [heatNum]
        mostDisadvantagedSkater = 0
        mostAdvantagedSkater = 0
        worstValue = self._laneAverage*float(self.heatSize)*100
        bestValue = 0
        allValues = []
        for skater_, value in skaterDict.items():
            allValues.append(sum(value['values'])/len(value['values']))
            if sum(value['values']) < worstValue:
                worstValue = sum(value['values'])
                mostDisadvantagedSkater = skater_
            if sum(value['values']) > bestValue:
                bestValue = sum(value['values'])
                mostAdvantagedSkater = skater_
        stddev = np.sqrt(
            sum([(x - self._laneAverage)**2 for x in allValues]))/len(allValues)

        lowestValueIndex = np.argmin(np.asarray(
            skaterDict[mostDisadvantagedSkater]['values']))
        correspondingHeat = skaterDict[mostDisadvantagedSkater]['heatNum'][lowestValueIndex]
        n_stddevIncreases = 0
        permitted_n_stddevIncreases = 50
        shift = -2
        heatDict_ = copy.copy(heatDict)
        while True:
            thisHeat = copy.copy(heatDict_[correspondingHeat]['heat'])
            skaterLoc = thisHeat.index(mostDisadvantagedSkater)
            while mostDisadvantagedSkater in thisHeat:
                thisHeat.remove(mostDisadvantagedSkater)
            while True:
                if skaterLoc + shift >= 0 and skaterLoc + shift < len(thisHeat):
                    thisHeat.insert(skaterLoc + shift, mostDisadvantagedSkater)
                    break
                shift = np.sign(shift)*(np.abs(shift) - 1)
            heatDict_[correspondingHeat]['heat'] = thisHeat
            skaterDict = {}
            for heatNum, heat in heatDict_.items():
                for lane, skater_ in enumerate(heat['heat']):
                    value = 0
                    if lane in self._laneValues.keys():
                        value = self._laneValues[lane]
                    if skater_ in skaterDict.keys():
                        skaterDict[skater_]['lanes'].append(lane+1)
                        skaterDict[skater_]['values'].append(value)
                        skaterDict[skater_]['heatNum'].append(heatNum)
                    else:
                        skaterDict[skater_] = {'values': [value]}
                        skaterDict[skater_]['heatNum'] = [heatNum]
                        skaterDict[skater_]['lanes'] = [lane+1]
            mostDisadvantagedSkater = 0
            mostAdvantagedSkater = 0
            worstValue = self._laneAverage*float(self.heatSize)*100
            bestValue = 0
            allValues = []
            for skater_, value in skaterDict.items():
                allValues.append(sum(value['values'])/len(value['values']))
                if sum(value['values']) < worstValue:
                    worstValue = sum(value['values'])
                    mostDisadvantagedSkater = skater_
                if sum(value['values']) > bestValue:
                    bestValue = sum(value['values'])
                    mostAdvantagedSkater = skater_
            newStddev = np.sqrt(
                sum([(x - self._laneAverage)**2 for x in allValues]))/len(allValues)
            shift = -2
            if newStddev < stddev:
                heatDict = heatDict_
                stddev = newStddev
                lowestValueIndex = np.argmin(np.asarray(
                    skaterDict[mostDisadvantagedSkater]['values']))
                correspondingHeat = skaterDict[mostDisadvantagedSkater]['heatNum'][lowestValueIndex]
                shift = -1
            elif newStddev == stddev:
                n_stddevIncreases += 1
                if n_stddevIncreases >= permitted_n_stddevIncreases:
                    break
                nextLowestValue = np.max(np.asarray(
                    skaterDict[mostDisadvantagedSkater]['values']))
                lowestValueIndex = np.argmax(np.asarray(
                    skaterDict[mostDisadvantagedSkater]['values']))
                lowestValue = np.min(np.asarray(
                    skaterDict[mostDisadvantagedSkater]['values']))
                for i, value in enumerate(skaterDict[mostDisadvantagedSkater]['values']):
                    if value < nextLowestValue and value != lowestValue:
                        nextLowestValue = value
                        lowestValueIndex = i

                correspondingHeat = skaterDict[mostDisadvantagedSkater]['heatNum'][lowestValueIndex]
            else:
                heatDict_ = copy.copy(heatDict)
                n_stddevIncreases += 1
                if n_stddevIncreases >= permitted_n_stddevIncreases:
                    break
                shift = 2
                mostDisadvantagedSkater = mostAdvantagedSkater
                lowestValueIndex = np.argmax(np.asarray(
                    skaterDict[mostDisadvantagedSkater]['values']))
                correspondingHeat = skaterDict[mostDisadvantagedSkater]['heatNum'][lowestValueIndex]

        if self.printDetails:
            self.startLaneLogger.info(
                'Assumed start lane values: %s', self._laneValues)
            self.startLaneLogger.info(
                'Stddev of start lane values: %s', newStddev)

        if verbose:
            print('Assumed start lane values: ', self._laneValues)
            print('Stddev of start lane values: ', newStddev)
        self.startLaneStddev = newStddev
        for skater_, value in skaterDict.items():
            self.skaterDict[skater_]._startPositionValues = value['values']
            self.skaterDict[skater_]._startLanes = value['lanes']
            if verbose:
                print('Skater {0} average start lane value: {1}, lanes: {2}'.format(skater_, sum(
                    value['values'])/len(value['values']), value['lanes']))
            if self.printDetails:
                aveVal = sum(value['values'])/len(value['values'])
                self.startLaneLogger.info(
                    'Skater %s average start lane value: %.3f , lanes: %s', skater_, aveVal, value["lanes"])

    def spaceHeatsOut(self, heatsDict: dict) -> list:
        """ Spaces out heats so that skaters aren't skating too often. """
        idealSpacing = self.totalSkaters // self.heatSize
        minimalSpacing = 2
        uniqueSkaters = []
        heatNums = []
        for heatNum, heat in heatsDict.items():
            heatNums.append(heatNum)
            for skater_ in heat['heat']:
                if skater_ in uniqueSkaters:
                    continue
                uniqueSkaters.append(skater_)
        concludedHeats = []
        concludedHeats_ = []
        n_attempts = -1
        while True:
            if n_attempts > 1000:
                concludedHeats_ = []
                break
            n_attempts += 1
            skatersConcluded = []
            n_heatsConcludedThisLoop = 0
            self.randomizer.shuffle(heatNums)
            for heatNum in heatNums:
                heat = heatsDict[heatNum]
                if heatNum in concludedHeats:
                    continue
                if any((x in skatersConcluded for x in heat['heat'])):
                    continue
                skatersConcluded += heat['heat']
                concludedHeats.append(heatNum)
                concludedHeats_.append(heatNum)
                n_heatsConcludedThisLoop += 1
                if ((len(skatersConcluded) == len(uniqueSkaters)) or
                        (n_heatsConcludedThisLoop >= idealSpacing)):
                    break
            if n_heatsConcludedThisLoop < minimalSpacing:
                concludedHeats = []
                concludedHeats_ = []
                continue
            if len(concludedHeats) == len(heatsDict):
                break
            concludedHeats_.append('Pause')
        return concludedHeats_

    def _gradOpt(self,
                 verbose: bool = True):
        """ Don't use this, it does not work yet """
        maxTheoreticalMatrixDims = (max(
            self.totalSkaters*(self.totalSkaters - 1)//4, self.numRacesPerSkater*2), self.heatSize*2)
        initM = initializeMatrix(self.totalSkaters,
                                 self.numRacesPerSkater,
                                 self.heatSize,
                                 maxTheoreticalMatrixDims)

        conTests = convergenceTests(minHeatSize=self.minHeatSize,
                                    verbose=verbose,
                                    printDetails=self.printDetails,
                                    averageSeeding=self.averageSeeding,
                                    numRacesPerSkater=self.numRacesPerSkater,
                                    sampleStdDev=self.sampleStdDev)
        for i in range(initM.shape[0]):
            if all((x == -1 for x in initM[i, :])):
                break
        # initM = initM[:i+1, :min(self.heatSize+3, self.heatSize*2)]
        initM = initM[:i+2, :]
        maxTheoreticalMatrixDims = initM.shape
        initPot = self.heatPotentialCalc(
            initM.flatten(), maxTheoreticalMatrixDims)
        print(initPot)
        n_encounterErrors = 0
        n_seedingErrors = 0
        n_appearancesErrors = 0
        shift = 1
        heatDict = None
        for n_attempts in range(100):
            if heatDict is not None:
                print(heatDict)
            direc = [0]*len(initM.flatten())
            upOrDown = [-1, 1]
            for i, elem in enumerate(initM.flatten()):
                shuffle(upOrDown)
                direc[i] = upOrDown[0]
                if elem <= 0:
                    direc[i] = 1
                if elem >= self.totalSkaters - 1:
                    direc[i] = -1
            direc = np.diag(direc)

            res = minimize(self.heatPotentialCalc,
                           initM.flatten(),
                           args=(maxTheoreticalMatrixDims,),
                           tol=100.0,
                           method='Powell',
                           options={'disp': True,
                                    'return_all': True,
                                    # 'direc': direc,
                                    'maxiter': 1000})
            out = np.rint(res.x.reshape(maxTheoreticalMatrixDims)).astype(int)
            initM = out
            col0 = initM[:, n_attempts % self.heatSize]
            np.random.shuffle(col0)
            initM[:, n_attempts % self.heatSize] = col0
            heatDict = {}
            for i, row in enumerate(out):
                outRow = row[(row >= 0) & (row < self.totalSkaters)]
                if len(outRow) > 0:
                    outRow = list(set(outRow.tolist()))
                    heatDict[i+1] = {}
                    heatDict[i+1] = {'heat': outRow}
                    heatDict[i+1]['averageSeeding'] = 0
                    for skaterNum in outRow:
                        if skaterNum in self.skaterDict.keys():
                            heatDict[i+1]['averageSeeding'] += self.skaterDict[skaterNum].seed
                    heatDict[i+1]['averageSeeding'] /= len(outRow)
            for heatNum, subDict in heatDict.items():
                heatAsList = subDict['heat']
                for skIndex, skNum in enumerate(heatAsList):
                    self.skaterDict[skNum].addHeatAppearance(heatNum)
                    for skNum_0 in heatAsList[skIndex+1:]:
                        self.skaterDict[skNum].addEncounterFlexible(skNum_0)
                        self.skaterDict[skNum_0].addEncounterFlexible(skNum)

            tr = conTests.heatLengthTest(heatDict,
                                         n_attempts,
                                         logger=self.buildHeatsLogger)
            if tr == 0:
                for skater_ in self.skaterDict.values():
                    skater_.removeAllHeatAppearances()
                    skater_.removeAllEncounters()
                continue

            tr, n_appearancesErrors = conTests.appearanceTest(n_appearancesErrors,
                                                              self.skaterDict,
                                                              n_attempts,
                                                              logger=self.buildHeatsLogger)
            if tr == 0:
                for skater_ in self.skaterDict.values():
                    skater_.removeAllHeatAppearances()
                    skater_.removeAllEncounters()
                continue
            n_attempts += 1
            tr, n_encounterErrors = conTests.encounterTest(n_encounterErrors,
                                                           shift,
                                                           self.skaterDict,
                                                           n_attempts,
                                                           logger=self.buildHeatsLogger)
            self.n_encounterErrors = n_encounterErrors
            if tr == 0:
                shift += 1
                for skater_ in self.skaterDict.values():
                    skater_.removeAllHeatAppearances()
                    skater_.removeAllEncounters()
                continue
            if self.considerSeeding:
                tr, n_seedingErrors = conTests.seedingTest(n_seedingErrors,
                                                           heatDict,
                                                           n_attempts,
                                                           logger=self.buildHeatsLogger)
                if tr == 0:
                    for skater_ in self.skaterDict.values():
                        skater_.removeAllHeatAppearances()
                        skater_.removeAllEncounters()
                    continue
            if self.fairStartLanes:
                self.makeStartLanesFair(heatDict)
            print('Success after {} attempts.'.format(n_attempts))
            if self.printDetails:
                self.buildHeatsLogger.info(
                    'Success after %s attempts.', n_attempts)
            return heatDict
        return {}
