from statsmodels.tsa.statespace.sarimax import SARIMAX

def sarimax_baseline(train, test):
    model = SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,1,7))
    results = model.fit(disp=False)
    return results.forecast(len(test))
