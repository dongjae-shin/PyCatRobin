import datetime

# calculate days between two days
def days_between(d1, d2):

    d1 = datetime.datetime.strptime(d1, "%Y-%m-%d")
    d2 = datetime.datetime.strptime(d2, "%Y-%m-%d")
    return abs((d2 - d1).days)

# show example
print(days_between("2020-01-01", "2020-01-02"))

# Q: what is variable?
# A: variable is a function that takes a string and returns a string
