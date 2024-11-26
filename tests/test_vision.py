import fairbench as fb

def test_utkface():
    y, yhat, sens = fb.bench.vision.utkface(classifier="flac", predict="predict")
    flac_report = fb.biasreport(predictions=yhat, labels=y, sensitive=fb.Fork(fb.categories @ sens))
    y, yhat, sens = fb.bench.vision.utkface(classifier="badd", predict="predict")
    badd_report = fb.biasreport(predictions=yhat, labels=y, sensitive=fb.Fork(fb.categories @ sens))
    report = fb.Fork(badd=badd_report, flac=flac_report)
    fb.text_visualize(report.maxrdiff)