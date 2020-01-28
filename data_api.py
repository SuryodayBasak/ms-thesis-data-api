import pandas as pd
import numpy as np
import os

"""
Dataset 1: The anuran calls dataset with only the family attribute used as
labels.
"""
class AnuranCallsFamily:
  def __init__(self):
    self.dataset = pd.read_csv('./data/anuran_calls/Frogs_MFCCs.csv')
  
  def Data(self):
    familyDataset = self.dataset.copy()
    familyDataset = familyDataset.drop(['Genus', 'Species', 'RecordID'],\
                    axis = 1)
    labels = {'Bufonidae':1, 'Dendrobatidae':2, 'Hylidae':3,\
              'Leptodactylidae': 4}
    familyDataset = familyDataset.replace({'Family': labels})
    familyDataset = familyDataset.to_numpy()
    X = familyDataset[:, :-1]
    y = familyDataset[:, -1]

    return X, y

"""
Dataset 2: The anuran calls dataset with only the Genus attribute used as
labels.
"""
class AnuranCallsGenus:
  def __init__(self):
    self.dataset = pd.read_csv('./data/anuran_calls/Frogs_MFCCs.csv')

  def Data(self):
    familyDataset = self.dataset.copy()
    familyDataset = familyDataset.drop(['Family', 'Species', 'RecordID'],\
                    axis = 1)
    labels = {'Adenomera': 1, 'Ameerega': 2, 'Dendropsophus': 3, \
              'Hypsiboas': 4, 'Leptodactylus': 5, 'Osteocephalus': 6, \
              'Rhinella': 7, 'Scinax': 8}
    familyDataset = familyDataset.replace({'Genus': labels})
    familyDataset = familyDataset.to_numpy()
    X = familyDataset[:, :-1]
    y = familyDataset[:, -1]

    return X, y

"""
Dataset 3: The anuran calls dataset with only the species attribute used as
labels.
"""
class AnuranCallsSpecies:
  def __init__(self):
    self.dataset = pd.read_csv('./data/anuran_calls/Frogs_MFCCs.csv')

  def Data(self):
    familyDataset = self.dataset.copy()
    familyDataset = familyDataset.drop(['Family', 'Genus', 'RecordID'],\
                    axis = 1)
    labels = {'AdenomeraAndre': 1, 'AdenomeraHylaedactylus': 2, \
              'Ameeregatrivittata': 3, 'HylaMinuta': 4, \
              'HypsiboasCinerascens': 5, 'HypsiboasCordobae': 6, \
              'LeptodactylusFuscus': 7, 'OsteocephalusOophagus': 8, \
              'Rhinellagranulosa': 9, 'ScinaxRuber': 10}
    familyDataset = familyDataset.replace({'Species': labels})
    familyDataset = familyDataset.to_numpy()
    X = familyDataset[:, :-1]
    y = familyDataset[:, -1]

    return X, y

"""
Dataset 4: The audit risk dataset.
"""
class AuditRisk:
  def __init__(self):
    self.dataset = pd.read_csv('./data/audit/audit_risk.csv')

  def Data(self):
    auditDataset = self.dataset.copy()
    auditDataset = auditDataset.to_numpy()
    X = auditDataset[:, :-1]
    y = auditDataset[:, -1]

    return X, y

"""
Dataset 5: The avila dataset.
"""
class Avila:
  def __init__(self):
    self.dataset = pd.read_csv('./data/avila/avila/avila-tr.txt', \
                   header = None)

  def Data(self):
    avilaDataset = self.dataset.copy()
    avilaDataset = avilaDataset.to_numpy()
    labels = {'A': 1, 'B': 2, 'C':3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8, \
              'I': 9, 'W': 10, 'X': 11, 'Y': 12}
    X = avilaDataset[:, :-1]
    y = avilaDataset[:, -1]
    y = np.array([labels[alphabet] for alphabet in y])

    return X, y

"""
Dataset 6: Banknote authentication.
"""
class BankNoteAuth:
  def __init__(self):
    self.dataset = np.genfromtxt('data/banknote/data_bnk.txt', delimiter = ',')
  
  def Data(self):
    return self.dataset[:, :-1], self.dataset[:, -1]

"""
Dataset 7: Blood transfusion.
"""
class BloodTransfusion:
  def __init__(self):
    self.dataset = pd.read_csv('data/blood_transfusion/transfusion.data')

  def Data(self):
    btDataset = self.dataset.copy()
    btDataset = btDataset.to_numpy()
    X = btDataset[:, :-1]
    y = btDataset[:, -1]

    return X, y

"""
Dataset 8: Breast cancer.
"""
class BreastCancer:
  def __init__(self):
    self.dataset = np.genfromtxt('data/breast_cancer/breast-cancer-wisconsin.data', \
                                  delimiter = ',')

  def Data(self):
    X = self.dataset[:, 1:-1]
    labels = {2: 0, 4: 1}
    y = self.dataset[:, -1]
    y = np.array([labels[n] for n in y])

    return X, y

"""
Dataset 9: Breast tissue.
"""
class BreastTissue:
  def __init__(self):
    self.dataset = pd.read_csv('data/breast_tissue/brr.csv', header = None)

  def Data(self):
    btDataset = self.dataset.copy()
    btDataset = btDataset.to_numpy()
    labels = {'car': 1, 'fad': 2, 'mas':3, 'gla': 4, 'con': 5, 'adi': 6}
    X = btDataset[:, 1:]
    y = btDataset[:, 0]
    y = np.array([labels[alphabet] for alphabet in y])

    return X, y

"""
Dataset 10: 
"""
class BurstHeaderPacket:
  def __init__(self):
    self.dataset = pd.read_csv('data/burst_header_packet/bhp.csv')

  def Data(self):
    bhpDataset = self.dataset.copy()
    nodeStatus = {'B': 1, 'NB': 2, 'P NB': 3}
    label = {'NB-No Block': 1, 'Block': 2, 'No Block': 3, 'NB-Wait': 4}
    bhpDataset = bhpDataset.replace({'Node Status': nodeStatus, \
                                     'Class': label})
    bhpDataset = bhpDataset.to_numpy()
    X = bhpDataset[:, :-1]
    y = bhpDataset[:, -1]
    
    return X, y

"""
Dataset 11:
"""
class CSection:
  def __init__(self):
    self.dataset = np.genfromtxt('data/c_section/caesarian.csv', \
                                 delimiter = ',')

  def Data(self):
    print(self.dataset)
    return self.dataset[:, :-1], self.dataset[:, -1]

"""
Dataset 12: Cardiooctography -- morphology.
"""
class CardioOtgMorph:
  def __init__(self):
    self.dataset = np.genfromtxt('data/cardioctography/ctg.csv', \
                                 delimiter = ',')

  def Data(self):
    print(self.dataset)
    return self.dataset[:, :-2], self.dataset[:, -2]

"""
Dataset 13: Cardiooctography -- fetal.
"""
class CardioOtgFetal:
  def __init__(self):
    self.dataset = np.genfromtxt('data/cardioctography/ctg.csv', \
                                 delimiter = ',')

  def Data(self):
    print(self.dataset)
    return self.dataset[:, :-2], self.dataset[:, -1]

