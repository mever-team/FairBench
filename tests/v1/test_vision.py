from fairbench import v1 as fb


def test_utkface_predict():
    y, yhat, sens = fb.bench.vision.utkface(classifier="flac", predict="predict")
    flac_report = fb.biasreport(
        predictions=yhat, labels=y, sensitive=fb.Fork(fb.categories @ sens)
    )
    y, yhat, sens = fb.bench.vision.utkface(classifier="badd", predict="predict")
    badd_report = fb.biasreport(
        predictions=yhat, labels=y, sensitive=fb.Fork(fb.categories @ sens)
    )
    report = fb.Fork(badd=badd_report, flac=flac_report)
    fb.text_visualize(report.maxrdiff)


def test_utkface_probabilities():
    y, yhat, sens = fb.bench.vision.utkface(classifier="flac", predict="probabilities")
    flac_report = fb.biasreport(
        predictions=yhat, labels=y, sensitive=fb.Fork(fb.categories @ sens)
    )
    y, yhat, sens = fb.bench.vision.utkface(classifier="badd", predict="probabilities")
    badd_report = fb.biasreport(
        predictions=yhat, labels=y, sensitive=fb.Fork(fb.categories @ sens)
    )
    report = fb.Fork(badd=badd_report, flac=flac_report)
    fb.describe(report.accuracy)


"""
def test_celeba():
    y, yhat, sens = fb.bench.vision.celeba(classifier="flac", predict="probabilities")
    flac_report = fb.biasreport(predictions=yhat, labels=y, sensitive=fb.Fork(fb.categories @ sens))
    y, yhat, sens = fb.bench.vision.celeba(classifier="mavias", predict="probabilities")
    mavias_report = fb.biasreport(predictions=yhat, labels=y, sensitive=fb.Fork(fb.categories @ sens))
    report = fb.Fork(mavias=mavias_report, flac=flac_report)
    fb.text_visualize(report.maxbarea)
"""


def test_watebirds():
    y, yhat, sens = fb.bench.vision.waterbirds(classifier="badd", predict="predict")
    badd_report = fb.biasreport(
        predictions=yhat, labels=y, sensitive=fb.Fork(fb.categories @ sens)
    )
    y, yhat, sens = fb.bench.vision.waterbirds(classifier="mavias", predict="predict")
    mavias_report = fb.biasreport(
        predictions=yhat, labels=y, sensitive=fb.Fork(fb.categories @ sens)
    )
    report = fb.Fork(mavias=mavias_report, badd=badd_report)
    fb.text_visualize(report.accuracy)
