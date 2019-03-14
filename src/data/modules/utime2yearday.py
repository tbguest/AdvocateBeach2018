
def utime2yearday(unixtime):
    # accepts a POSIX timestamp (seconds since 1970 epoch) and converts to yearday with decimal seconds
    # TBGuest, 5 Mar 2019

    from datetime import datetime

    jnk = datetime.fromtimestamp(unixtime)
    yr = int(jnk.strftime("%Y"))

    dt = datetime(yr, 1, 1)
    yearday = (np.array(unixtime) - time.mktime(dt.timetuple()))/86400

    return yearday
