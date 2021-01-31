for seg in range(len(joints)):
    if np.size(joints[seg][0][0]) == 0 or np.size(joints[seg][1][0]) == 0:
        joints[seg] = "#"
for seg in range(len(joints)):
    if joints[seg] != "#":
        toCheck = joints[seg][0][0]
        tally = 0
        for i in range(len(toCheck)):
            if joints[toCheck[i]] == "#":
                tally += 1
        if tally == len(toCheck):
            joints[seg] = "#"
        else:
            toCheck = joints[seg][1][0]
            for i in range(len(toCheck)):
                if joints[toCheck[i]] == "#":
                    tally += 1
            if tally == len(toCheck):
                joints[seg] = "#"
print(joints)
for i in range(len(joints)):
    if joints[i] != "#":
        plt.plot(splitRoads[i][0], splitRoads[i][1])
        plt.show()