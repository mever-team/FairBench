try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import auc, roc_curve
    from sklearn.model_selection import train_test_split
except ModuleNotFoundError:
    print("sklearn not found (it is not a mandatory dependency): when needed, FairBench will resort to custom numpy implementations")
    from fairbench.bench.fallbacks.learning.logistic_regression import LogisticRegression
    from fairbench.bench.fallbacks.learning.min_max_scaler import MinMaxScaler
    from fairbench.bench.fallbacks.learning.auc import auc, roc_curve
    from fairbench.bench.fallbacks.read_csv import train_test_split

try:
    from pandas import read_csv, get_dummies, concat
except ModuleNotFoundError:
    print("pandas not found (it is not a mandatory dependency): when needed, FairBench will resort to a simpler CSV reader")
    from fairbench.bench.fallbacks.read_csv import read_csv, get_dummies, concat

