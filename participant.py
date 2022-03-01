# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 20:21:28 2022

@author: rasta
"""


class skater():
    
    def __init__(self, 
                 skaterNum: int, 
                 seed : int = 0, 
                 team : None or str = 'Wildcats',
                 ageCategory : str = 'senior',
                 name : None or str = 'sk8r'):
        self.encounters = []
        self.skaterNum = skaterNum
        self.heatAppearances = []
        self.totalAppearances = 0
        self.totalEncounters = 0
        self.totalUniqueEncounters = 0
        self.cumulativeEncounters = 0
        self.points = 0
        self.averageResult = 0
        self.rating = 0
        self.seed = seed
        self.team = team
        self.ageCategory = ageCategory
        self.name = name
        self._startPositionValues = []
        self._startLanes = []
    
    def addEncounter(self, otherSkaterNum : int):
        if otherSkaterNum != self.skaterNum and not(otherSkaterNum in self.encounters):
            self.encounters.append(otherSkaterNum)
        self.totalEncounters = len(self.encounters)
        self.totalUniqueEncounters = len(set(self.encounters))
    
    def removeEncounter(self, otherSkaterNum : int):
        while otherSkaterNum in self.encounters:
            self.encounters.remove(otherSkaterNum)
        self.totalEncounters = len(self.encounters)
        self.totalUniqueEncounters = len(set(self.encounters))

    def addEncounterFlexible(self, otherSkaterNum : int):
        if otherSkaterNum != self.skaterNum:
            self.encounters.append(otherSkaterNum)
        self.totalEncounters = len(self.encounters)
        self.totalUniqueEncounters = len(set(self.encounters))
    
    def removeEncounterFlexible(self, otherSkaterNum : int):
        if otherSkaterNum in self.encounters:
            self.encounters.remove(otherSkaterNum)
            self.totalEncounters = len(self.encounters)
            self.totalUniqueEncounters = len(set(self.encounters))
    
    def addHeatAppearance(self, heatNum : int):
        if not(heatNum in self.heatAppearances):
            self.heatAppearances.append(heatNum)
            self.totalAppearances = len(self.heatAppearances)
    
    def removeHeatAppearance(self, heatNum : int):
        while heatNum in self.heatAppearances:
            self.heatAppearances.remove(heatNum)
        self.totalAppearances = len(self.heatAppearances)

    def averageResults(self):
        if self.totalEncounters != 0:
            self.averageResult = self.points / self.totalEncounters
        else:
            self.averageResult = 0.0
    
    def updateRunningAverageResult(self, n_encounters: int):
        self.cumulativeEncounters += n_encounters
        if self.cumulativeEncounters != 0:
            self.rating = self.points / self.cumulativeEncounters
        else:
            self.rating = 0.0
        