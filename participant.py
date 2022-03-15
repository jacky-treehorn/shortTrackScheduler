# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 20:21:28 2022

@author: rasta
"""
# pylint: disable=invalid-name
import numpy as np


class skater():
    """ Creates a skater for a distance in a short track competetion. """

    def __init__(self,
                 skaterNum: int,
                 seed: int = 0,
                 team: None or str = 'Wildcats',
                 ageCategory: str = 'senior',
                 name: None or str = 'sk8r'):
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
        self.heatTimes = {}
        self.bestTime = float(np.iinfo(int).max)

    def addEncounter(self, otherSkaterNum: int):
        """ Adds a different skater to this skater's encounters. """
        if otherSkaterNum != self.skaterNum and not (otherSkaterNum in self.encounters):
            self.encounters.append(otherSkaterNum)
        self.totalEncounters = len(self.encounters)
        self.totalUniqueEncounters = len(set(self.encounters))

    def removeEncounter(self, otherSkaterNum: int):
        """ Removes a different skater from this skater's encounters. """
        while otherSkaterNum in self.encounters:
            self.encounters.remove(otherSkaterNum)
        self.totalEncounters = len(self.encounters)
        self.totalUniqueEncounters = len(set(self.encounters))

    def removeAllEncounters(self):
        self.encounters = []
        self.totalEncounters = len(self.encounters)
        self.totalUniqueEncounters = len(set(self.encounters))

    def addEncounterFlexible(self, otherSkaterNum: int):
        """ Flexibly adds a different skater to this skater's encounters. """
        if otherSkaterNum != self.skaterNum:
            self.encounters.append(otherSkaterNum)
        self.totalEncounters = len(self.encounters)
        self.totalUniqueEncounters = len(set(self.encounters))

    def removeEncounterFlexible(self, otherSkaterNum: int):
        """ Flexibly removes a different skater from this skater's encounters. """
        if otherSkaterNum in self.encounters:
            self.encounters.remove(otherSkaterNum)
            self.totalEncounters = len(self.encounters)
            self.totalUniqueEncounters = len(set(self.encounters))

    def addHeatAppearance(self, heatNum: int):
        """ Adds a heat to this skater's heats. """
        if not (heatNum in self.heatAppearances):
            self.heatAppearances.append(heatNum)
            self.totalAppearances = len(self.heatAppearances)

    def removeHeatAppearance(self, heatNum: int):
        """ Removes a heat from this skater's heats. """
        while heatNum in self.heatAppearances:
            self.heatAppearances.remove(heatNum)
        self.totalAppearances = len(self.heatAppearances)

    def removeAllHeatAppearances(self):
        """ Removes all heats from this skater's heats. """
        self.heatAppearances = []
        self.totalAppearances = len(self.heatAppearances)

    def averageResults(self):
        """ Generates an overall performance of the skater. """
        if self.totalEncounters > 0:
            self.averageResult = self.points / self.totalEncounters
        else:
            self.averageResult = 0.0

    def updateRunningAverageResult(self, n_encounters: int):
        """ Generates an intermediate performance of the skater. """
        self.cumulativeEncounters += n_encounters
        if self.cumulativeEncounters != 0:
            self.rating = self.points / self.cumulativeEncounters
        else:
            self.rating = 0.0

    def addHeatTime(self, heatNum: int, time: float = -1.0):
        """ Adds a time to this skater's times. """
        self.heatTimes[heatNum] = time

    def calculateBestTime(self):
        """ Gets this skater's best time. """
        for val in self.heatTimes.values():
            if val <= 0.0:
                continue
            if val < self.bestTime:
                self.bestTime = val
