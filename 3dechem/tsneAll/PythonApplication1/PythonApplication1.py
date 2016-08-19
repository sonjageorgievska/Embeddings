import fileinput #for reading large files
import json
import random
import numpy as np
import os
import shutil
import csv 
import math
from datetime import datetime
import argparse as argp
from decimal import Decimal
import scipy.sparse as spar
import sklearn.manifold as mani
import py_compile
import tables as tb

py_compile.compile('PythonApplication1.py')

#region parse command-line args
parser = argp.ArgumentParser(
   description = 'This script creates input files for CCluster hierarchical cluster visualization tool.')

parser.add_argument(
   '-i',
   dest = 'graphFile',
   help = 'Input file containing a similarity graph in edge list format')

parser.add_argument(
   '-c',
   dest = 'clustFile',
   help = 'Input file containing hierarchical clusters')

parser.add_argument(
   '-m',
   dest = 'metaFile',
   help = 'Input file containing cluster meta data')

parser.add_argument(
   '-d',
   dest = 'baseDir',
   help = 'Base directory to store output files')

parser.add_argument(
   '-p',
   dest = 'precision',
   help = 'precision of embedding (between 1 and 30)')

args = parser.parse_args()
#endregion
           
#class Embedding:

#region Reading data


def ReadMetaDataFile(metaDataFile):
    metaDataDict = dict()
    numberOfPoints = 50 #452292
    f = tb.open_file('F:/3dEchemalldata/dist_all.packedfrozen.h5', 'r')
    a = f.root.scores        
    X = [a[i] for i in range(numberOfPoints)]           
    for i in range(numberOfPoints):
        metaDataDict[i] = X[i]
    return metaDataDict

def ReadPropertiesIntensitiesFile(propertiesIntensitiesFile):
    """File format: [id] [intensityOfProperty1] [intensityOfProperty2]... [intensityOfPropertyN]"""      
    intensitiesDict = dict()
    for line in fileinput.input([propertiesIntensitiesFile]):
        if line != "\n":   
            items = line.split()
            id = items[0]     
            items.pop(0)                             
            intensitiesDict[id] = items
    return intensitiesDict


def ReadMatrix():
    numberOfPoints = 50 #452292
    f = tb.open_file('F:/3dEchemalldata/dist_all.packedfrozen.h5', 'r')
    a = f.root.scores[:numberOfPoints][:numberOfPoints]      
    maxim = 65535  
    epsilon = 0.01
    #X=a     
    X = [[max(-math.log2((a[i][j] + epsilon)/maxim),0) for j in range(numberOfPoints)] for i in range(numberOfPoints)]        
    maximum = 0 
    for i in range(numberOfPoints):
        for j in range(numberOfPoints):  
            maximum = max(maximum, max(X[i]))    
    X = [[X[i][j]/maximum for j in range(numberOfPoints)] for i in range(numberOfPoints)]
    return X

#endregion 
    
#region Analytics
          

def ComputeCoordinates(keys):
    """Uses the tSNE algorithm to determine and fix the coordinates of objects with ids in keys.
    """
    fixedCoordinate = dict()
    if len(keys) > 1:
        tsne = mani.TSNEDISC(n_components=3,  random_state=0,  metric = 'precomputed')
        X = ReadMatrix()     
        print(str(datetime.now()) + ": Start tsne...")
        Y = tsne.fit(X) 
        print(str(datetime.now()) + ": Finished tsne...")
        coord = Y.embedding_
        maxim = 0
        for i in range(len(coord)):
            for j in range (3):
                maxim = max(maxim, math.fabs(coord[i][j]))
        for key in keys:            
            a = coord[int(key)][0]/maxim
            b=  coord[int(key)][1]/maxim
            c=  coord[int(key)][2]/maxim
            fixedCoordinate[key] = [a,b,c]
    return fixedCoordinate    

#region Write output


def CreateSmallDataJSONFile(allPoints, startingFolder):
    string = json.dumps(allPoints)
    file = open(os.path.join(startingFolder, "smalldata.json"), "w")
    file.write(string)
    file.close()
   
def CreateMetaDataFileForBigDataMode(startingFolder, bigdatamode):
    string = "var bigData =" + bigdatamode + ";"
    file = open(os.path.join(startingFolder, "MetaData.js"), "w")
    file.write(string)
    file.close()

def CreateTrivialPathsDictionary(keys):
    pathsDict = dict()
    for key in keys:
        pathsDict[key] = ["0"]
    return pathsDict

def CreatePointsDictionary(fixedCoordinates, pathsDict, metaDataDict, intensitiesOfPropertiesDict):
    pointsDict = dict()
    
    for key in fixedCoordinates.keys():
        point = dict()

        point["Path"] = pathsDict[key]
        point["Coordinates"] = fixedCoordinates[key]
        #point["Coordinates"].append(0)
        if (metaDataDict != "no" ):
            if key in metaDataDict:
                point["Categories"] = metaDataDict[key]
            else:
                point["Categories"] = []
        else: 
            point["Categories"] = []
        if (intensitiesOfPropertiesDict != "no"):                
            point["Properties"] = intensitiesOfPropertiesDict[key]
        else: 
            point["Properties"] = []
        pointsDict[key] = point
    return pointsDict

def CreateDirIfDoesNotExist(dirname):
    if not os.path.exists(dirname):          
        os.makedirs(dirname)

def RemoveDirTreeIfExists(dirname):        
    if os.path.exists(dirname):
        shutil.rmtree(dirname)

            
#endregion 

#region Workflow

def ConvertCoordinatesToList(fixedCoordinate):
    for key in fixedCoordinate:
        fixedCoordinate[key] = list(fixedCoordinate[key])
                       
def Workflow(similarityMatrixFile, metaDataFile, namesOfPropertiesFile, propertiesIntensitiesFile, baseDir):      
    print(str(datetime.now()) + ": Removing old data...")
    dirname1 =  os.path.join(baseDir, "data")
    RemoveDirTreeIfExists(dirname1)
    print(str(datetime.now()) + ": Reading input files...")
    if metaDataFile != "No":
        metaDataDict = ReadMetaDataFile(metaDataFile)
    else: 
        metaDataDict = "no"
    if propertiesIntensitiesFile != "No":
        intensitiesDict = ReadPropertiesIntensitiesFile(propertiesIntensitiesFile)
    else:
        intensitiesDict = "no"   
        
    pathsDict = CreateTrivialPathsDictionary(metaDataDict.keys())
    print(str(datetime.now()) + ": Start ..")      
    fixedCoordinate = ComputeCoordinates(metaDataDict.keys())
    ConvertCoordinatesToList(fixedCoordinate)   
    pointsDict = CreatePointsDictionary(fixedCoordinate, pathsDict, metaDataDict, intensitiesDict)        
    print(str(datetime.now()) + ": Start writing output...")         
    CreateDirIfDoesNotExist(dirname1)
    CreateSmallDataJSONFile(pointsDict, dirname1)
    shutil.copy(namesOfPropertiesFile, dirname1)
    CreateMetaDataFileForBigDataMode(dirname1, false)
    print(str(datetime.now()) + ": Finished writing output.")
#endregion
         
#region Main

Workflow("validatedtypeyeastITS_OptSimBiolomics.txt", "validatedtypeyeastITS_Species.txt", "NamesOfProperties.json","No", os.getcwd())

#Workflow(args.graphFile, args.clustFile, args.metaFile, "NamesOfProperties.json","No", args.baseDir)
#endregion