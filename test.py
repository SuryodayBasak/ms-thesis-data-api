import data_api as da

audit = da.CardioOtgFetal()
X, y = audit.Data()
print(X, y)

