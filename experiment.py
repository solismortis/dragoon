import datetime


now = datetime.datetime.now()
not_now = now + datetime.timedelta(hours=25)
print(not_now)
print(datetime.timedelta(hours=25))