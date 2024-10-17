import numpy as np
from numpy import genfromtxt

class NetworkGenerator():
    def __init__(self, path):
        self.path = path

    def GenerateRandomNetwork(self, N, r):
        nodes = []

        for k in range(N):
            l = np.random.uniform(0, 1)
            m = np.random.uniform(0, 1)
            n = Node(l, m)
            nodes.append(n)

        #for n in nodes:
            #print("n.i", n.i, "n.j", n.j)

        #print("start edge generation")
        adj = np.zeros((N, N), dtype=np.int16)
        for i in range(N):
            for j in range(i + 1, N):
                xi = nodes[i].i
                yi = nodes[i].j
                xj = nodes[j].i
                yj = nodes[j].j
                dist = np.sqrt(np.square(xi - xj) + np.square(yi - yj))
                if dist <= r:
                    adj[i][j] = 1
                #print("xi", xi, "yi", yi, "xj", xj, "yj", yj, "dist", dist, "r", r, "i", i, "j", j, "adj[i][j]", adj[i][j])
        # print(adj)
        return adj

    def WriteNetwork(self, name, adj):
        filename = self.path
        filename += name + ".npy"
        np.save(filename, adj)

    def WriteNetwork2(self, name, adj):
        filename = self.path
        filename += name + ".txt"
        f1 = open(filename, 'a')
        length = len(adj)
        for i in range(length):
            for j in range(length):
                element = str(adj[i][j]) + ", "
                f1.write(element)
            f1.write("\n")

    def ReadNetwork(self, Name):
        adj = np.load(Name + ".npy")
        return adj

    def ReadNetwork2(self, name):
        filename = "E:\\aaht14\\OneDrive\\OneDrive - Institute of Business Administration\\TD\\CodeBSP_20191129\\cases\\"
        filename += name + ".txt"
        adj = genfromtxt(filename, delimiter=',', dtype=int)
        return adj

    def GenerateCase(self, case, N, r):
        print("start case", case, "N", N, "r", r)
        adj = self.GenerateRandomNetwork(N, r)
        self.WriteNetwork2(case, adj)
        print("end case", case, "N", N, "r", r)

    def GenerateAllCases(self):
        # case 1
        self.GenerateCase("case1r1", 100, 1 / np.sqrt(100))
        self.GenerateCase("case1r2", 100, 1 / np.sqrt(100))
        self.GenerateCase("case1r3", 100, 1 / np.sqrt(100))

        # case 2
        self.GenerateCase("case2r1", 300, 1 / np.sqrt(300))
        self.GenerateCase("case2r2", 300, 1 / np.sqrt(300))
        self.GenerateCase("case2r3", 300, 1 / np.sqrt(300))

        # case 3
        self.GenerateCase("case3r1", 500, 1 / np.sqrt(500))
        self.GenerateCase("case3r2", 500, 1 / np.sqrt(500))
        self.GenerateCase("case3r3", 500, 1 / np.sqrt(500))

        # case 4
        self.GenerateCase("case4r1", 750, 1 / np.sqrt(750))
        self.GenerateCase("case4r2", 750, 1 / np.sqrt(750))
        self.GenerateCase("case4r3", 750, 1 / np.sqrt(750))

        # case 5
        self.GenerateCase("case5r1", 1000, 1 / np.sqrt(1000))
        self.GenerateCase("case5r2", 1000, 1 / np.sqrt(1000))
        self.GenerateCase("case5r3", 1000, 1 / np.sqrt(1000))

        # case 6
        self.GenerateCase("case6r1", 100, 2 / np.sqrt(100))
        self.GenerateCase("case6r2", 100, 2 / np.sqrt(100))
        self.GenerateCase("case6r3", 100, 2 / np.sqrt(100))

        # case 7
        self.GenerateCase("case7r1", 300, 2 / np.sqrt(300))
        self.GenerateCase("case7r2", 300, 2 / np.sqrt(300))
        self.GenerateCase("case7r3", 300, 2 / np.sqrt(300))

        # case 8
        self.GenerateCase("case8r1", 500, 2 / np.sqrt(500))
        self.GenerateCase("case8r2", 500, 2 / np.sqrt(500))
        self.GenerateCase("case8r3", 500, 2 / np.sqrt(500))

        # case 9
        self.GenerateCase("case9r1", 750, 2 / np.sqrt(750))
        self.GenerateCase("case9r2", 750, 2 / np.sqrt(750))
        self.GenerateCase("case9r3", 750, 2 / np.sqrt(750))

        # case 10
        self.GenerateCase("case10r1", 1000, 2 / np.sqrt(1000))
        self.GenerateCase("case10r2", 1000, 2 / np.sqrt(1000))
        self.GenerateCase("case10r3", 1000, 2 / np.sqrt(1000))

        # case 11
        self.GenerateCase("case11r1", 100, 3 / np.sqrt(100))
        self.GenerateCase("case11r2", 100, 3 / np.sqrt(100))
        self.GenerateCase("case11r3", 100, 3 / np.sqrt(100))

        # case 12
        self.GenerateCase("case12r1", 300, 3 / np.sqrt(300))
        self.GenerateCase("case12r2", 300, 3 / np.sqrt(300))
        self.GenerateCase("case12r3", 300, 3 / np.sqrt(300))

        # case 13
        self.GenerateCase("case13r1", 500, 3 / np.sqrt(500))
        self.GenerateCase("case13r2", 500, 3 / np.sqrt(500))
        self.GenerateCase("case13r3", 500, 3 / np.sqrt(500))

        # case 14
        self.GenerateCase("case14r1", 750, 3 / np.sqrt(750))
        self.GenerateCase("case14r2", 750, 3 / np.sqrt(750))
        self.GenerateCase("case14r3", 750, 3 / np.sqrt(750))

        # case 15
        self.GenerateCase("case15r1", 1000, 3 / np.sqrt(1000))
        self.GenerateCase("case15r2", 1000, 3 / np.sqrt(1000))
        self.GenerateCase("case15r3", 1000, 3 / np.sqrt(1000))

        # case 16
        self.GenerateCase("case16r1", 100, 4 / np.sqrt(100))
        self.GenerateCase("case16r2", 100, 4 / np.sqrt(100))
        self.GenerateCase("case16r3", 100, 4 / np.sqrt(100))

        # case 17
        self.GenerateCase("case17r1", 300, 4 / np.sqrt(300))
        self.GenerateCase("case17r2", 300, 4 / np.sqrt(300))
        self.GenerateCase("case17r3", 300, 4 / np.sqrt(300))

        # case 18
        self.GenerateCase("case18r1", 500, 4 / np.sqrt(500))
        self.GenerateCase("case18r2", 500, 4 / np.sqrt(500))
        self.GenerateCase("case18r3", 500, 4 / np.sqrt(500))

        # case 19
        self.GenerateCase("case19r1", 750, 4 / np.sqrt(750))
        self.GenerateCase("case19r2", 750, 4 / np.sqrt(750))
        self.GenerateCase("case19r3", 750, 4 / np.sqrt(750))

        # case 20
        self.GenerateCase("case20r1", 1000, 4 / np.sqrt(1000))
        self.GenerateCase("case20r2", 1000, 4 / np.sqrt(1000))
        self.GenerateCase("case20r3", 1000, 4 / np.sqrt(1000))


class Node():
    def __init__(self, i, j):
        self.i = i
        self.j = j
