import fairbench as fb

x_train, y_train, x, y, train, test = fb.bench.tabular.adult(predict="data")
print("Entries", x_train.shape)
