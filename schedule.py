# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 20:41:04 2022

@author: rasta
"""
import numpy as np
from participant import skater
from random import Random
import copy

class heat():
    
    def __init__(self):
        pass
        
class raceProgram():
    
    def __init__(self, 
                 totalSkaters: int = 0,
                 numRacesPerSkater : int = 0,
                 heatSize: int = 2,
                 minHeatSize: int = 2,
                 considerSeeding : bool = False,
                 fairStartLanes : bool = False,
                 participantNames : list = []
                 ):
        self.participantNames = participantNames
        if len(self.participantNames) != 0:
            assert len(self.participantNames) == totalSkaters, 'Length of participant names does not match the number of skaters!'
        self.heats = []
        self.totalSkaters = totalSkaters
        self.numRacesPerSkater = numRacesPerSkater
        self.heatSize = heatSize
        assert self.heatSize > 1, 'Heat size must be at least 2.'
        self.considerSeeding = considerSeeding
        self.fairStartLanes = fairStartLanes
        self.minHeatSize = minHeatSize
        assert self.minHeatSize >= 1, 'Minimum heat size must be at least 1.'
        assert self.heatSize >= self.minHeatSize, 'Heat size must be at least {}.'.format(self.minHeatSize)
        self.skaterDict = {}
        self.randomizer = Random()
        self._laneValues = {0:6.424, 1:3.527, 2:2.401, 3:1.374, 4:1.0, 5:1.0, 6:1.0, 7:1.0, 8:1.0, 9:1.0, 10:1.0}
        self._laneAverage = 0.0
        for i, (lane, val) in enumerate(self._laneValues.items()):
            self._laneAverage += val
            if i + 1 >= self.heatSize:
                break
        self._laneAverage /= float(self.heatSize)
        self.heatOrder = []
        self.heatDict = {}
    
    def buildHeats(self, 
                   max_attempts: int = 10000,
                   adjustAfterNAttempts: int = 500,
                   encounterFlexibility: int = 0,
                   verbose: bool = False) -> dict:
        adjustAfterNAttempts = min(max_attempts, adjustAfterNAttempts)
        if self.numRacesPerSkater == 0:
            while self.totalSkaters / 2**self.numRacesPerSkater > 4:
                self.numRacesPerSkater += 1
        else:
            assert self.numRacesPerSkater > 0, 'numRaces must be greater than 0.'
        skaterNums = list(range(self.totalSkaters))
        averageSeeding = float(self.totalSkaters + 1) / 2.0  
        sampleStdDev = np.std(skaterNums)/np.sqrt(self.heatSize)   
        for i_skater in skaterNums:
            skaterName = 'Person_'+str(i_skater)
            if i_skater < len(self.participantNames):
                skaterName = self.participantNames[i_skater]
            self.skaterDict[i_skater] = skater(i_skater, 
                                               seed = i_skater + 1,
                                               name = skaterName)
        heatDict = {}
        n_attempts = 1
        n_encounterErrors = 0
        n_personalEncounterErrors = 0
        n_seedingErrors = 0
        n_appearancesErrors = 0
        n_appearanceErrorsResets = 0
        shift = 0
        while True:
            n_attempts += 1
            if n_appearancesErrors > adjustAfterNAttempts:
                n_appearancesErrors = 0
                n_appearanceErrorsResets += 1
                if n_appearanceErrorsResets % 2:
                    n_encounterErrors = 0
                    shift += 1
                    print('Increasing shift: {}'.format(shift))
                else:
                    n_personalEncounterErrors = 0
                    encounterFlexibility += 1
                    print('Increasing encounterFlexibility: {}'.format(encounterFlexibility))
            if n_encounterErrors > adjustAfterNAttempts:
                n_encounterErrors = 0
                shift += 1
                print('Increasing shift: {}'.format(shift))
            if n_personalEncounterErrors > adjustAfterNAttempts:
                n_personalEncounterErrors = 0
                encounterFlexibility += 1
                print('Increasing encounterFlexibility: {}'.format(encounterFlexibility))
            if n_seedingErrors > adjustAfterNAttempts:
                n_seedingErrors = 0
                sampleStdDev *= 1.1
                print('Increasing sampleStdDev: {}'.format(sampleStdDev))
            if n_attempts >= max_attempts:
                print('No success after {} attempts. Quitting.'.format(n_attempts))
                return {}
            self.randomizer.shuffle(skaterNums)
            for i_skater in skaterNums:
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
                        self.skaterDict[i_skater].addEncounterFlexible(otherSkaterNum)
                        self.skaterDict[otherSkaterNum].addEncounterFlexible(i_skater)
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
                        self.skaterDict[skaterNum_0].removeHeatAppearance(heatNum)
                        for skaterNum_1 in heat['heat']:
                            if skaterNum_1 != skaterNum_0:
                                self.skaterDict[skaterNum_0].removeEncounterFlexible(skaterNum_1)
                                self.skaterDict[skaterNum_1].removeEncounterFlexible(skaterNum_0)
            for heatNum in heatsToDelete:
                del heatDict[heatNum]
            if not all([len(heat_['heat']) >= self.minHeatSize for heat_ in heatDict.values()]):
                print('\n')
                print('heatSizeError: Attempt {0} produced an unfavourable Heat structure, modifying...\n'.format(n_attempts))
                self.reorganizeHeats(heatDict)
            if all([skater_.totalAppearances == self.numRacesPerSkater for skater_ in self.skaterDict.values()]):
                allEncounters = [x.totalEncounters for x in self.skaterDict.values()]
                encountersError = False
                for i in range(len(allEncounters)):
                    for j in range(i+1, len(allEncounters)):
                        if np.abs(allEncounters[i] - allEncounters[j]) > shift:
                            encountersError = True
                            break
                if encountersError:
                    n_encounterErrors += 1
                    if verbose:
                        print('encountersError: Attempt {0} produced an unfavourable Heat structure, modifying...\n'.format(n_attempts))
                    n_attempts += 1
                    self.reorganizeHeats(heatDict)
                    continue
                seedingErrors = False
                if self.considerSeeding:
                    for heatNum, heat in heatDict.items():
                        if np.abs(heat['averageSeeding'] - averageSeeding) > sampleStdDev:
                            seedingErrors = True
                if seedingErrors:
                    n_seedingErrors += 1
                    print('seedingErrors: Attempt {0} produced an unfavourable Heat structure, modifying...\n'.format(n_attempts))
                    self.reorganizeHeats(heatDict)
                    continue        
                print('Success after {} attempts.'.format(n_attempts))
                break
            else:
                if verbose:
                    print('\n')
                    print('totalAppearancesError: Attempt {0} produced an unfavourable Heat structure, modifying...\n'.format(n_attempts))
                n_appearancesErrors += 1
                self.reorganizeHeats(heatDict)
        if shift > 1:
            print('\n')
            print('WARNING! Some skaters may have noticably fewer encounters than others.')
            print('\n')
        if self.fairStartLanes:
            self.makeStartLanesFair(heatDict)
        for heatNum, heat in heatDict.items():
            if self.considerSeeding:
                print('Heat {0}: '.format(heatNum), heat['heat'], ' Seeding Check: {0}'.format(np.abs(heat['averageSeeding'] - averageSeeding) < sampleStdDev))
            else:
                print('Heat {0}: '.format(heatNum), heat['heat'])
        heatSpacing = self.spaceHeatsOut(heatDict)
        if len(heatSpacing) > 0:
            print('\n')
            print('Heats should be run in the following order: ', heatSpacing)
            self.heatOrder = [x for x in heatSpacing if type(x) == int]
            heatDict_ = {}
            for i, heat in enumerate(self.heatOrder):
                heatDict_[i] = heatDict[heat]
            heatDict = heatDict_   
        self.heatDict = heatDict
        for skaterNum, skater_ in self.skaterDict.items():
                print('Skater {0} appears in {1} heats: '.format(skater_.skaterNum, skater_.totalAppearances), skater_.heatAppearances, ', Total encounters: {0}, Total unique encounters: {1}'.format(skater_.totalEncounters, skater_.totalUniqueEncounters))       
            
        return heatDict

    def reorganizeHeats(self, heatDict: dict):
        removedSkaterDict = {}
        for heatNum, heat in heatDict.items():
            if len(heat['heat']) >= min(3, self.heatSize):
                removedSkaterIndex = self.randomizer.randint(0, len(heat['heat']) - 1)
                removedSkaterDict[heatNum] = heat['heat'][removedSkaterIndex]
        for heatNum, skaterNum in removedSkaterDict.items():
            self.skaterDict[skaterNum].removeHeatAppearance(heatNum)
            for skaterNum_0 in heatDict[heatNum]['heat']:
                if skaterNum_0 != skaterNum:
                    self.skaterDict[skaterNum].removeEncounter(skaterNum_0)
                    self.skaterDict[skaterNum_0].removeEncounter(skaterNum)
            while skaterNum in heatDict[heatNum]['heat']:
                heatDict[heatNum]['heat'].remove(skaterNum)      
            
    def makeStartLanesFair(self, heatDict):
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
                    skaterDict[skater_] = {'values':[value]}
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
        stddev = np.mean(allValues)     
            
        lowestValueIndex = np.argmin(np.asarray(skaterDict[mostDisadvantagedSkater]['values']))
        correspondingHeat = skaterDict[mostDisadvantagedSkater]['heatNum'][lowestValueIndex]   
        n_stddevIncreases = 0
        permitted_n_stddevIncreases = 50
        shift = -2
        while True:
            heatDict_ = copy.copy(heatDict)    
            thisHeat = copy.copy(heatDict_[correspondingHeat]['heat'])
            skaterLoc = thisHeat.index(mostDisadvantagedSkater)
            while mostDisadvantagedSkater in thisHeat:
                thisHeat.remove(mostDisadvantagedSkater)
            while True:
                if skaterLoc + shift >= 0 and skaterLoc + shift < len(thisHeat):
                    thisHeat.insert(skaterLoc + shift, mostDisadvantagedSkater)
                    break
                else:
                    shift = np.sign(shift)*(np.abs(shift) - 1)
            heatDict_[correspondingHeat]['heat'] = thisHeat
            skaterDict = {}
            for heatNum, heat in heatDict_.items():
                for lane, skater_ in enumerate(heat['heat']):
                    value = 0
                    if lane in self._laneValues.keys():
                        value = self._laneValues[lane]
                    if skater_ in skaterDict.keys():
                        skaterDict[skater_]['values'].append(value)
                        skaterDict[skater_]['heatNum'].append(heatNum)
                    else:
                        skaterDict[skater_] = {'values':[value]}
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
            newStddev = np.std(allValues)
            shift = -2
            if newStddev < stddev:
                heatDict = heatDict_
                stddev = newStddev
                lowestValueIndex = np.argmin(np.asarray(skaterDict[mostDisadvantagedSkater]['values']))
                correspondingHeat = skaterDict[mostDisadvantagedSkater]['heatNum'][lowestValueIndex]
                shift = -1
            elif newStddev == stddev:
                n_stddevIncreases += 1
                if n_stddevIncreases >= permitted_n_stddevIncreases:
                    break
                nextLowestValue = np.max(np.asarray(skaterDict[mostDisadvantagedSkater]['values']))
                lowestValueIndex = np.argmax(np.asarray(skaterDict[mostDisadvantagedSkater]['values']))
                lowestValue = np.min(np.asarray(skaterDict[mostDisadvantagedSkater]['values']))
                for i, value in enumerate(skaterDict[mostDisadvantagedSkater]['values']):
                    if value < nextLowestValue and value != lowestValue:
                        nextLowestValue = value
                        lowestValueIndex = i
                        
                correspondingHeat = skaterDict[mostDisadvantagedSkater]['heatNum'][lowestValueIndex]
            else:
                n_stddevIncreases += 1
                if n_stddevIncreases >= permitted_n_stddevIncreases:
                    break    
                shift = 2
                mostDisadvantagedSkater = mostAdvantagedSkater
                lowestValueIndex = np.argmax(np.asarray(skaterDict[mostDisadvantagedSkater]['values']))
                correspondingHeat = skaterDict[mostDisadvantagedSkater]['heatNum'][lowestValueIndex]
        print('\n')
        print('Assumed start lane values: ',self._laneValues)
        print('Stddev of start lane values: ', newStddev)
        for skater_, value in skaterDict.items():
            print('Skater {0} average start lane value: '.format(skater_), sum(value['values'])/len(value['values']))
        print('\n')

    def spaceHeatsOut(self, heatsDict: dict) -> list:
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
        self.randomizer.shuffle(heatNums)
        concludedHeats = []
        concludedHeats_ = []
        n_attempts = -1
        while True:
            if n_attempts > 1000:
                print('WARNING!: No suitable heat spacing could be found after 1000 attempts.')
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
                if any([x in skatersConcluded for x in heat['heat']]):
                    continue
                skatersConcluded += heat['heat']
                concludedHeats.append(heatNum)
                concludedHeats_.append(heatNum)
                n_heatsConcludedThisLoop += 1
                if len(skatersConcluded) == len(uniqueSkaters) or n_heatsConcludedThisLoop >= idealSpacing:
                    break  
            if n_heatsConcludedThisLoop < minimalSpacing:
                concludedHeats = []
                concludedHeats_ = []
                continue
            if len(concludedHeats) == len(heatsDict):
                break
            concludedHeats_.append('Pause')
        return concludedHeats_