import datetime

NOW = datetime.datetime.now()

DATA_DIRECTORY = "./data/" + str(NOW.year) + "-" + str(NOW.month) + "-" + str(NOW.day)

PHONE_DIRECTORY = "./data/" + str(NOW.year) + "-" + str(NOW.month) + "-" + str(NOW.day) + "/phonelist"