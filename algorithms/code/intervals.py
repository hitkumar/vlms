from typing import List


class Interval:
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __repr__(self):
        return f"Interval({self.start}, {self.end})"


class Problems:
    def canAttendMeetings(self, intervals: List[Interval]) -> bool:
        intervals.sort(key=lambda x: x.start)
        for i in range(1, len(intervals)):
            prev_interval = intervals[i - 1]
            curr_interval = intervals[i]
            if curr_interval.start < prev_interval.end:
                return False

        return True

    def minMeetingRooms(self, intervals: List[Interval]) -> int:
        start_times = sorted([i.start for i in intervals])
        end_times = sorted([i.end for i in intervals])
        count, res = 0, 0
        s, e = 0, 0
        while s < len(start_times):
            if start_times[s] < end_times[e]:
                s += 1
                count += 1
            else:
                e += 1
                count -= 1
            res = max(res, count)
        return res

    def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
        intervals.sort(key=lambda x: x[0])
        end = intervals[0][1]
        res = 0

        for i in range(1, len(intervals)):
            curr_interval = intervals[i]
            if curr_interval[0] < end:
                end = min(end, curr_interval[1])
                res += 1
            else:
                end = max(end, curr_interval[1])
        return res


def main():
    intervals = [
        Interval(1, 3),
        Interval(2, 6),
        Interval(8, 10),
        Interval(15, 18),
        Interval(6, 17),
    ]
    intervals.sort(key=lambda x: x.start)
    print(intervals)


if __name__ == "__main__":
    main()
