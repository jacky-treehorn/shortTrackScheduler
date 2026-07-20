@echo off

set EXE_PATH="%~dp0skatingScheduler.exe"

:: Free testing, without a winsport input file
rem --method can be sgp, minimize or random (random has not been tested recently though)
rem --runSimulation if true will run a little competition simulation, which may be interesting
rem --winsportInput if given will take "VAR08",4,"Number of Races" as --numRacesPerSkater
rem                 "VAR01",1,"Current Event" as --winsportEventName and
rem 				"VAR14","S" <-- Helmet number,"Name","FNam","Team","TeamAbr","NatTm","STAT","RankW","RankD","ACRNR","Q" <-- Seeding,"G","F","RC","Time","Rank","CDR","FP","SP","SupFin","Vic","Time"
rem 				The helmet number will be added as an extra column to each row of --winsportOutputFullPath
rem					I do not know if this is the intention of the input file, but we can change things at any time.
%EXE_PATH% ^
  --totalSkaters 22 ^
  --numRacesPerSkater 4 ^
  --heatSize 4 ^
  --considerSeeding false ^
  --fairStartLanes true ^
  --minHeatSize 3 ^
  --method minimize ^
  --runSimulation false ^
  --winsportOutputFullPath "C:\Users\rasta\gitRepos\shortTrackScheduler\dist\winsport.txt" ^
  --winsportEventName "superDuperEvent"

pause

:: Using a winsport input file

%EXE_PATH% ^
  --heatSize 4 ^
  --considerSeeding false ^
  --fairStartLanes true ^
  --minHeatSize 3 ^
  --method sgp ^
  --runSimulation false ^
  --winsportOutputFullPath "C:\Users\rasta\gitRepos\shortTrackScheduler\dist\winsport_2.txt" ^
  --winsportInput "C:\Users\rasta\gitRepos\shortTrackScheduler\winsportInput.txt"

pause